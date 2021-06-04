from gps_babel_tower.tasks.nearest_neighbor import NearestNeighbor
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Models: https://huggingface.co/models?search=dpr
nn = NearestNeighbor(model_id='facebook/dpr-ctx_encoder-single-nq-base',
                     data_file_path='nn_sample.txt',
                     index_file_path='/tmp/nn_index.faiss')

print(nn.get_nearest_neighbors(query='iPhone 12', top_k=3))
print(nn.get_nearest_neighbors(query='xbox one', top_k=3))




