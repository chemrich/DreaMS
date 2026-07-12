import h5py
import numpy as np
import pandas as pd
# import nndescent
# import pynndescent
import ngtpy
from tqdm import tqdm
from pathlib import Path


def load_gems_embs(pth, logger):
    # TODO: This is a temporary fix, we need to get rid of blank samples in a better way
    with h5py.File(pth, 'r') as f:
        # Get rid of "blank" samples
        gemc_C_names = pd.Series(f['name'][:]).astype(str)
        logger.info(f'Num. of spectra: {gemc_C_names.shape}')
        blank_substrs = ['blank', 'wash', 'no_inj', 'noinj', 'empty', 'solvent']
        gemc_C_names = gemc_C_names[gemc_C_names.apply(lambda n: all([s not in n.lower() for s in blank_substrs]))]
        idx_gems = gemc_C_names.index.tolist()
        logger.info(f'Num. of non-blank spectra: {gemc_C_names.shape}')
        # Get embeddings
        gemc_C_embs = f['DreaMS_Embedding'][:][idx_gems]
    return gemc_C_embs


def iter_embs(lib_pth, gems_pths, logger):
    """Yield embedding chunks in exactly the order they are inserted into the NGT index.

    Called twice: once to build the index, once to query it. Object id `i` in the index
    therefore corresponds to the i-th vector yielded here, counting across all chunks.
    """
    logger.info(f'Loading embeddings from {lib_pth}.')
    with h5py.File(lib_pth, 'r') as f:
        yield f['DreaMS_embedding'][:].astype(np.float16)

    for p in gems_pths:
        logger.info(f'Loading embeddings from {p}.')
        embs = load_gems_embs(p, logger)

        num_nans = np.count_nonzero(np.isnan(embs))
        logger.info(f'Num. of NaNs: {num_nans}.')
        if num_nans > 0:
            embs = np.nan_to_num(embs)
            logger.info(f'Num. of NaNs after replacing: {np.count_nonzero(np.isnan(embs))}.')

        yield embs.astype(np.float16)


def build_knn_graph(ngt_index, emb_chunks, k, num_embs, tqdm_logger=None):
    """Query the index for the k nearest neighbours of every object, excluding itself.

    `emb_chunks` must re-yield the embeddings in insertion order, because the query vector
    for object `i` has to be supplied by the caller: `ngt_index.get_object(i)` CANNOT be
    used to recover it. On an index built with object_type='Float16' (as this one is),
    get_object returns corrupt — often all-zero — vectors, and a zero vector has no cosine
    neighbours, so search() then returns an empty list. Verified against ngt 2.3.12.
    """
    knn_i, knn_j, knn_w = [], [], []

    i = 0
    pbar = tqdm(total=num_embs, desc='Constructing k-NN graph', file=tqdm_logger)
    for embs in emb_chunks:
        for emb in embs:
            res = np.array(ngt_index.search(emb, k + 1))
            # Drop self by object id rather than assuming it ranks first: NGT search is
            # approximate, and duplicate embeddings are common in the spectral libraries.
            res = res[res[:, 0] != i][:k]
            nns, dists = res.T
            sims = 1 - dists  # 'Cosine' distance -> cosine similarity

            knn_i.extend([i] * len(nns))
            knn_j.extend(nns.astype(np.int64))
            knn_w.extend(sims)
            i += 1
            pbar.update()
    pbar.close()

    if i != num_embs:
        raise RuntimeError(f'Queried {i} embeddings but the index holds {num_embs}.')

    return np.array(knn_i), np.array(knn_j), np.array(knn_w)


def main():
    # Imported lazily: only main() needs msml, so iter_embs/build_knn_graph stay importable
    # (and testable) without pulling in the heavy checkpoint-unpickling dependency.
    from msml.utils.io import setup_logger, TqdmToLogger

    k = 3
    # lib_pth = Path('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_A_merged_spectra_dreams.pkl')
    lib_pth = Path('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_merged_spectra_dreams.hdf5')
    gems_dir = Path('/storage/plzen1/home/romanb/msvn_C')
    out_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    name = f'DreaMS_Atlas_{k}NN_ngt_float16'
    out_pth = out_dir / f'{name}.npz'
    logger = setup_logger(out_pth.with_suffix('.log'))
    tqdm_logger = TqdmToLogger(logger)

    gems_pths = list(gems_dir.glob('msvn_C_H1000_KK1.*.hdf5'))
    gems_pths = sorted(gems_pths, key=lambda p: int(p.name.split('.')[-2]))  # Sort by chunk ids

    with h5py.File(lib_pth, 'r') as f:
        dim = f['DreaMS_embedding'].shape[1]

    # Create NGT index
    logger.info('Creating NGT index.')
    ngtpy.create(
        str(out_dir / f'{name}'),
        dimension=dim,
        distance_type='Cosine',
        object_type='Float16',
        edge_size_for_creation=30,
        edge_size_for_search=60
    )
    ngt_index = ngtpy.Index(str(out_dir / f'{name}'))

    logger.info('Inserting embeddings into NGT index.')
    for embs in tqdm(iter_embs(lib_pth, gems_pths, logger), desc='Adding chunks to NGT index',
                     file=tqdm_logger):
        ngt_index.batch_insert(embs)
        logger.info('Saving NGT index.')
        ngt_index.save()

    logger.info('Constructing k-NN graph.')
    knn_i, knn_j, knn_w = build_knn_graph(
        ngt_index,
        iter_embs(lib_pth, gems_pths, logger),  # second pass, same order
        k,
        ngt_index.get_num_of_objects(),
        tqdm_logger,
    )

    logger.info('Saving k-NN graph.')
    np.savez(out_pth, i=knn_i, j=knn_j, w=knn_w)

    logger.info('Done.')


if __name__ == '__main__':
    main()
