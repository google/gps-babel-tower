*NOTE*  "Locally" in this doc refers to running a python command manually for debugging purpose (as opposed to submitting a docker to AI platform).

# Running Locally
Note you need to have GPU in order to run. It's recommended to run this command on Jupyter instances on GCP.
To run the example locally, run the following command:
```
python app/text2image.py --output_path=gs://gps-babel-tower/output --use_auth_token=true
```
Please log into huggingface if you specify `use_auth_token=true`.
If you want to write the output images to a local path, just set `--output_path=/tmp`.
If you want to run it in a conda environment you can also call
```
conda run -n babel python app/text2image.py --output_path=gs://gps-babel-tower/output --use_auth_token=true
```

# Build the Docker Image
To build the docker image, run the following command:
```
PROJECT_ID=$(gcloud config get project) ./build_docker.sh
```
This will build the docker image and push onto your GCP project.


# Running the Docker Image Locally
You need to use `nvidia-docker` to be able to use GPU. It's recommended to run this command on Jupyter instances on GCP.
```
nvidia-docker run gcr.io/$(gcloud config get project)/babel_example:v1 python /app/text2image.py --output_path=gs://gps-babel-tower/output
```

# Submitting to AI Platform
Use the following command:
```
PROJECT_ID=$(gcloud config get project)  ./run_on_ai_platform.sh --args="python,/app/text2image.py,--output_path=gs://gps-babel-tower/output"
```
After submitting the job, you can see the job status in the Vertex AI training page for custom jobs.