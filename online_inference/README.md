ML project API
====================
There are 2 options to create docker image for the application:
1. To build image on local machine, go to project root directory where Dockerfile is located and execute:
~~~
docker build -t image_name .
~~~
2. To download image from hub:
~~~
docker pull aershov3/ml_prod:image_name
~~~
To run a container based on the image:
~~~
docker run -d --name container_name -p 8000:8000 image_name
~~~
---------------------
Once started, add /docs in the url to see the list of available commands:
- /get-user/{user_id} - returns information on the user with the provided user_id
- /create-user/{user_id} - adds user to the dictionary
- /predict/{user-id} - predicts heart disease condition (positive - disease, negative - no disease) 
- /health/{config_name} - runs an application health check provided train parameters configuration file name (config_name)
----------------------
To generate predictions for user with the given user_id:
~~~
python main/predict_user.py user_id
~~~
-----------------------
To run inference tests:
~~~
pytest
~~~
-----------------------
Docker image optimisation
-----------------------
To optimise the docker image size I have tried to reduce the number of files that are copied from the project working directory to the image file 
and the number of packages that are used to build the image. I have created 3 docker images (image1, image2 and image3) and pushed them to a remote repository: https://hub.docker.com/r/aershov3/ml_prod/tags.
The first one (image1) was created by copying everything from the current directory to the image file (COPY . .). 
image2 was created by only copying modules that are used during inference. As a result, the compressed image size was reduced by approximately 300 MB. 
Finally, I reduced the number of packages to the required minimum in requirements.txt file  and rebuilt the image (image3). This resulted in further 100 MB decrease in image size.  