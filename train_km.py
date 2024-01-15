import numpy as np
from pathlib import Path
import argparse
import joblib
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

parser = argparse.ArgumentParser()
parser.add_argument("--segment_path", type=str, default='/data/common/LibriSpeech/segments/sdhubert_base')
parser.add_argument("--save_path", type=str, default='km')
parser.add_argument("--save_name", type=str, default=None)
parser.add_argument("--train_file", type=str, default='files/librispeech_train_10Ksubset.txt')
parser.add_argument("--n_clusters", type=int, default=16384)
parser.add_argument("--n_clusters_agglomerative", type=int, default=4096)

if __name__ == '__main__':
    args = parser.parse_args()
    segment_path = Path(args.segment_path)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    save_name = args.save_name if args.save_name is not None else segment_path.stem
    km_path = save_path/f'{save_name}_{args.n_clusters}.pt'
    reducer_path = save_path/f'{save_name}_{args.n_clusters}to{args.n_clusters_agglomerative}.npy'
    
    with open(args.train_file, 'r') as f:
        train_files = [f.rstrip() for f in f.readlines()]
    segment_files = [segment_path/file.replace('.flac','.npy') for file in train_files]
    segment_files.sort()
    
    print(f'{save_name}: {len(segment_files)} files are found.')
    feats = np.concatenate([np.load(file, allow_pickle=True)[()]['segment_features'] for file in segment_files],0)
    
    print(f'Features: {feats.shape}')
    if Path(km_path).exists() and not retrain_km:
        print(f'Loading pretrained KMeans model...')
        km_model=joblib.load(km_path)
    else:
        print(f'Training KMeans with {args.n_clusters} clusters...')
        km_model = MiniBatchKMeans(
                n_clusters=args.n_clusters,
                init="k-means++",
                max_iter=100,
                batch_size=10000,
                verbose=1,
                compute_labels=False,
                tol=0.0,
                max_no_improvement=100,
                init_size=None,
                n_init=5,
                reassignment_ratio=0.0,
            )
        km_model.fit(feats)
        joblib.dump(km_model, km_path)
        
    print(f'Merging cluster centers {args.n_clusters}-->{args.n_clusters_agglomerative}...')
    agg_cls = AgglomerativeClustering(args.n_clusters_agglomerative, affinity='euclidean', linkage='ward').fit_predict(km_model.cluster_centers_)
    np.save(reducer_path, agg_cls)
        
    