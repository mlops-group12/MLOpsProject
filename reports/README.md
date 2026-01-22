# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [x] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [x] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

12

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

*s181487, s214613, s214644, s206759*

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We haven't used third-party frameworks! Our project is build around pytorch and pytorch-lightning. 

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used uv to manage both our Python environment and project dependencies. This tool provides a consistent and efficient workflow across all team members and ensures that the project can be easily reproduced by others. Dependency versions are recorded in a uv.lock file, which is committed to the GitHub repository. This lock file guarantees that everyone installs the exact same versions of all packages, ensuring reproducibility across systems.
To get an identical development environment, a new team member simply needs to clone the repository and run uv sync, which installs all dependencies based on the lock file (given they have installed and are using uv as their package/env manager). 

Alternatively, if using pip, they can install dependencies with pip install -r requirements.txt. The requirements.txt file is created from the uv.lock file using uv export --format requirements.txt, which can also be installed using uv via uv pip install -r requirements.txt.
New dependencies can be added using uv add <package-name>. Team members can verify that the lock file is up to date with uv lock --check, and update dependency versions when necessary using uv lock --upgrade. 

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized the project using the provided cookiecutter template and followed its overall structure. We populated the main folders, including src, configs, dockerfiles, and tests, while keeping the template’s organization intact. Within the src directory, we added a frontend.py script to handle the api's frontend-related functionality. We also added an evaluate.dockerfile to the dockerfiles directory for evaluation purposes, a base_config file in configs to store shared configuration settings, and integration tests in the tests directory.
These additions were made to better separate responsibilities within the project, improve readability, and support smoother collaboration among group members by reducing merge conflicts.
We deviated slightly from the template by removing the data folder. Instead of storing data locally, our src/data.py module loads data directly from Google Cloud Storage as the dataset is too large to be committed to the repository. For reproducibility, we also included functionality in the data loader to download the dataset directly from Kaggle for users who wish to run the project independently.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

What we implemented: 

Following good coding practices is important in larger projects because it improves understandability when reviewing code written by others or when returning to older parts of the project. Since developers often have very individual coding styles, clear structure and documentation help bridge the gap between different approaches and make it easier to understand what specific functions, scripts, or modules are intended to do.
Additionally, using a standardized project structure, such as a cookiecutter template, helps reduce code duplication and prevents directory-related errors. It furthermore helps maintain an overview of the code structure. Overall, these practices reduce the time and effort required to understand, maintain, and extend code written by others or earlier in the project.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented a total of 7 unittests testing different parts of the data and the model. For the data we tested the batchsize after dataloading and the size of the dataset. for the model we tested if the model can run a forward step and each of training, test and validation step. Additionally we tested the optimizer. We also created some implementation tests, testing the dataloading and the frontend API.

test_data.py : Testing the length of the data and that the get_datasets function returns the expected datasets. It also tests for batch shape.

test_model.py : This unittest tests the model framework. This includes testing the shapes after the forward pass, and testing whether or not the training, validation and test step outputs the expected. Finally, the setup is configured to use a single optimizer in the pytorch-lightning framework, so this is also tested for.


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

we have a total code coverage of 79% but most of the important parts of the code is covered.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In this project, we initially set up individual branches for group members to become acquainted with the codebase and avoid conflicts on the main branch. As the project progressed, we also created branches for specific tasks, such as data version control, connecting to Google Cloud Storage, configuring the Cloud Build file, and writing unit and integration tests. This approach reduced merge conflicts by clearly separating work according to task.

All team members were set as repository owners, and pull requests were managed by reviewing the branch history using the Git Graph module. We discussed changes within the group before merging branches into the main branch, ensuring that updates were integrated safely. This process was repeated regularly throughout project days, preventing the accumulation of large conflicts and maintaining a stable main branch.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

DVC was included in this project as a learning tool rather than for its functionality in our specific implementation. However, DVC would be highly valuable in industrial or research settings where data are continuously updated. When data points are added or removed and model predictions change, DVC allows for reproduction of earlier experiments to identify the cause of the changed predictions and verify that they originate from data modifications rather than code changes. DVC is also beneficial when existing data are updated or processed through multiple stages. In such cases, changes in preprocessing can lead to errors or differences in model performance. By versioning each data state, DVC makes it possible to trace back through previous data versions and identify the root cause of such issues.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

Currently our continuous integration is build upon two different workflows:
- a pytest workflow
- a linting workflow

and we have also added a dependabot workflow to allow for updates to the environment. 

The pytest workflow runs on multiple os systems, specifically ubuntu, windows and macos (the latest available versions).

Both the pytest and linting workflow uses caching to speed up the continuous pipeline.

Here is an example:



## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured experiments using a base_config.yaml file that defines hyperparameters such as the random seed, number of epochs, learning rate and device. Using a config file ensures reproducibility and automatic logging of all settings. The configuration is also integrated with Weights & Biases (wandb), so each run and its parameters are tracked.
Experiments are run via the command line, with optional overrides, for example: **python -m src.mlopsproject.train wandb.enabled=true**


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Reproducibility was ensured by consistently logging both model artifacts and experiment configurations. Each experiment produced a trained model that was saved as a .pt file in a dedicated models file. In this project, we were not focused on comparing multiple hyperparameter configurations, so only the latest model was saved. However, in a setting where model comparison is important, the model filename would include configuration details such as learning rate, seed, or epoch count to avoid information loss.
To further secure reproducibility, experiments could be run using a fixed base_config.yaml file, ensuring that hyperparameters, random seeds, and device settings remained consistent across runs. When Weights & Biases (wandb) was enabled, all experiment details, including configuration parameters, metrics, and training logs, were automatically tracked and stored. This allows any experiment to be fully reproduced by simply reusing the logged configuration and rerunning the training script with the same settings.
Together, model saving, configuration files, and experiment tracking ensured that no critical information was lost and that experiments could be reliably reproduced.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

Our configuration file contains several hyperparameters, however in this experiment, we focus on comparing different values for the learning rate and the number of epochs.
First, we compare training the model using different learning rates, as shown in this [Figure](figures/sweep_lr.png). We evaluate learning rates of $[3×10^{-4},5×10^{-4},7×10^{-4}]$ , all trained for 10 epochs. From the figure, we observe no significant differences in performance among the three learning rates, all of the curves follow echother closly ending with a validation accucarcy at around 0.65 and validation loss just under 0.9. Therefore, when testing the number of epochs, we select the middle learning rate of $5×10^{-4}$. Nevertheless, evaluating the learning rate is important because it determines how efficiently and stably the model learns during training. An inappropriate learning rate can lead to slow convergence or unstable training, even if the model architecture is well chosen.

Next, we examine the number of epochs to identify a “sweet spot” where performance continues to improve without unnecessary additional training that yields no significant gains. As shown in the [Figure](figures/sweep_epoch.png), the performance begins to plateau around step 15k, which corresponds to approximately 20 epochs. However, increasing the number of epochs may still lead to minor performance improvements, but these gains may not justify the additional computational cost. After around 20 epochs, the model appears to begin diverging slowly.

In addition to the hyperparameters considered here, one could also investigate other factors, such as batch size or model hyperparameters, including the number of layers or the number of units per layer.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project, we developed separate Dockerfiles for training, evaluation, the API, and the frontend. We build our Docker images via Google Cloud run, which triggers when we push to the main branch of the repo. We used Docker to ensure that our applications can run on any PC, as they run on a virtual machine with the same settings. Here is a link to the GitHub location of our train Dockerfile https://github.com/mlops-group12/MLOpsProject/blob/main/dockerfiles/train.dockerfile. To run the training docker image: "docker run --rm europe-west1-docker.pkg.dev/active-premise-484209-h0/my-container-repo/train:latest" which will run the latest training file in the google cloud artifact registry.
 

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

During code development, we inevitably encountered errors and bugs. To minimize their impact, we used error messages and warnings in appropriate places. For example, a ValueError was raised when attempting to load a model that could not be found. Additionally, when errors occurred, as they often do during development, we frequently used print statements to verify that the program was running as expected and to help locate the source of the issue. 
The [Figure](figures/profile.png) shows the profiling results for our training script. From this, we can identify that the data loader is the most time-consuming task. Although the time per call is relatively low, it is invoked nearly 6,000 times, resulting in a high total runtime. In contrast, saving checkpoints takes roughly twice as long per call, but since it is only executed eight times, it contributes far less to the overall runtime.


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

Our Container registry can be seen in [Figure](figures/registry.png), where we have 4 containers api, evaluate, frontend and train.

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train our model in the cloud using vertex AI. We did this by creating a custom job which used a training docker image we had stored in the artifact registry. We configured the job to run on a CPU-based machine because our Docker images aren’t set up to handle GPU support. Thus, the cloud training didn’t provide any actual advantage to running locally in terms of speed. As a result, most experiments were conducted locally, while Vertex AI was only used to test if our cloud infrastructure worked correctly.
Vertex AI was chosen because it provides a managed interface for running custom containers, integrates seamlessly with Artifact Registry, and allows training jobs to be launched without managing underlying infrastructure.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We did manage to write an API for our emotion detection model using FastAPI. The API exposes a simple interface where an image is uploaded and an emotion prediction is returned. The model is loaded once at application startup, which avoids reloading the model for each request and improves performance. To make the deployment more realistic, the trained model is automatically downloaded from Google Cloud Storage (GCS), ensuring that the latest available model is always used.
For inference, the uploaded image is preprocessed to match the training setup. The image is converted to grayscale, resized to 64×64 pixels, and transformed into a PyTorch tensor before being passed to the model. The predicted emotion is selected as the class with the highest probability.
We added a GET / endpoint to verify that the API is running and a POST /predict endpoint for performing predictions. Additionally, the API returns the model version used for inference, improving transparency and reproducibility of predictions.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We managed to deploy our API both locally and in the cloud. The first step was local deployment, where we loaded the model from local files. The next step was loading the model from Google Cloud Storage (GCS), which required some additional setup, but we eventually succeeded. By creating a Dockerfile for the API, we were able to build a container image for our backend and later also for our frontend. Using the Google Cloud interface, we deployed the container to Cloud Run, which created a service and exposed it through a public URL pointing to the API backend. Using this URL and the /docs endpoint, we were able to test our model and run predictions.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

Group member s181487 used x, group member s214613 used x, group member s214644 used x and group member s206759 used x. So in total x credits was spend during the course and project.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

The [Figure](figures/architecture.png) describe the overall architecture of our system.

The central element of our system architecture is the developer, who can choose to work either locally or in the cloud. Locally, the developer can run our face detection model for training or evaluation, both of which are configurable through config files that log the chosen model setup, with optional logging to Weights & Biases (WandB). The output of training is a model that can be used by an API, which connects to a frontend interface where users can upload images and receive emotion predictions.

The given image input details are stored to enable later analysis of data drift 

In the cloud workflow, the developer begins by committing code to GitHub, which triggers GitHub Actions workflows. These workflows first perform unit tests, linting, and integration tests to ensure code quality. If the commit is on the main branch, the workflow continues by building new Docker images, which are stored in the Artifact Registry. The registry contains four images: the training and evaluation images, which run on Vertex AI to create models, logging relevant information to WandB, with the trained models and data stored in a cloud bucket that developers can access using DVC; and the API and frontend images, which can be deployed separately. Once the training is complete, the API image can be deployed with the trained model, providing an endpoint for predictions. After that, the frontend image can be deployed, configured to use the API URL to create an interface where users can upload images and receive emotion predictions. 

In addition, developers can always choose to clone the GitHub repository to work locally or modify the workflow.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
