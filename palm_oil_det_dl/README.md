#steps [`linux/mac users`] \
(1) download kaggle using -- `pip install kaggle` \
\
(2) go to your kaggle page login if you have an \
    account or create one if you don't then login \
\
(3) after login, you have to create your kaggle terminal \
    login credentials by going to `your profile`, \
    then click on `account` and then on `create new api token`.\
\
(4) by now the token should be downloaded then put \
    this token in this folder `~/.kaggle/` \
\
(5) now you have to change the permission of the \
    file by using `chmod 600 /home/vcode/.kaggle/kaggle.json` \
\
(6) the next thing to do is, go to your dataset\
    folder from your terminal and run this \
    command `kaggle competitions download -c widsdatathon2019` \
\
(7) once download is done unzip using the following \
    series of command still in the same dataset folder.\
    (7a) unzip widsdatathon2019.zip \
    (7b) unzip leaderboard_holdout_data.zip \
    (7c) unzip leaderboard_test_data.zip \
    (7d) unzip train_images.zip \
    (7e) 
\
(8) cd to your project directory \
\
(9) install `pipenv` using `pip3 install --user pipenv` \
(10) to create or activate the environment type `pipenv shell`\
(11) once environment is activated, you can install the required packages using `pipenv install`\
(12) download miniconda from `https://docs.anaconda.com/anaconda/install/uninstall/`\
(13) install miniconda using `bash TheNameOfTheMiniconda.sh` for linux. for me it is `Miniconda3-py39_4.9.2-Linux-x86_64.sh`\
(14) create new environment using `conda create -n palmoil_detection_gpu python=3.8 numpy pandas matplotlib -y`\
(15) activate the environment using `conda activate palmoil_detection_gpu`\
(16) sudo apt-get install python3-tk
*— — — — — — — — Possible Error at this point: — — — — — — — —*\
`conda: command not found`\

#### This occurs because the path for anaconda installation has not been set in your .bashrc or .zshrc\

### *Try:*\
`export PATH="/home/username/anaconda3/bin:$PATH"`\

*— — — — — — — — -End of the Error Resolving — — — — — — — —*\


*— — — — — — — — If it works and then error persist like this — — — — — — — —*\
`conda: command not found`\

#### This occurs because the path for anaconda installation has not been set in your .bashrc or .zshrc permanently\

### *Try:*\
`gedit ~/.profile`\

Add the line. `export PATH="/home/username/anaconda3/bin:$PATH"`\
Log out and log in again.
*— — — — — — — — -End of the Error Resolving — — — — — — — —*

(16) run `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y` to install pytorch with gpu.
(17) if you have a gpu and you have linux run `sudo apt install nvidia-cuda-toolkit` to install package to enable gpu

###### Break out

(palm_oil_detection) vcode@vcodePC:~/Documents/palm_oil_detection$ git init
hint: Using 'master' as the name for the initial branch. This default branch name
hint: is subject to change. To configure the initial branch name to use in all
hint: of your new repositories, which will suppress this warning, call:
hint: 
hint:   git config --global init.defaultBranch <name>
hint: 
hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
hint: 'development'. The just-created branch can be renamed via this command:
hint: 
hint:   git branch -m <name>
Initialized empty Git repository in /home/vcode/Documents/palm_oil_detection/.git/

| Details                   |                        |
|---------------------------|------------------------|
| Programming Language:     |  Python 3.**6**        |
| pytorch version:          |  >=1.7.1               |
| pytorch lightning version |  >=0.8.5               |
| Models Required:          | trained resnet50 model |
| OS                        | Ubuntu 18


#Short introduction to the project
This is a project that aims to help detect Diebetic Rethinopathy and it stage, the result of the project is a model that has learnt to differentiate between the various stages of the disease.

## Project Set Up and Installation on Ubuntu
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

[step 1]
The project heavily depends on python as it's the language used to run the whole program so, the first thing to do is to install python before proceeding.

[step 2]
install pip, pip is a python package that helps install other python packages.

[step 3]
use pip to install the rest of the required package. To do this, make sure your current directory structure is this from the terminal or command line.
.
├── dataset
│   |──Images
|   |    |──test_folder
|   |
|   └──labels
|         |──csv_file
|
├── models
|   └──model_file
|
|── __init__.py
|── .gitignore
|── DR_note.ipynb
|── process_data.py
├── README.md
├── requirements.txt
|── test.py
|── train.py
|── WRITEUP.md

type, 
**pip install -r requirements.txt** in the terminal

this command will install other required package for the project.

#model
This project also depends on just one AI model, to run which is the result of the training of a pretrained resnet50 model. This model will be together with the file.

#how to use the process_data.py script
In cases where the data csv file has 5 classes, The process_data.py script is used to re-label the script, to use this file, csv file must contain these two columns, image_id and target, image_id been the image name and target been the label.

#demo run process_data.py
use the python3 process_data.py -lp dataset/labels/test1.csv -lo dataset/labels/test_new1.csv -sd True -cl True

where **dataset/labels/test1.csv** should be the path to your dataset csv file and **dataset/labels/test_new1.csv** should be path where you want to save it and the name you want to save it with.

#Write on how to use train.py script
to run a new training you can use this code
reset && python3 train.py -tdp dataset/labels/test_new2.csv -tdr dataset/Images/test2/ -vdr dataset/Images/test2/ -vdp dataset/labels/test_new2.csv -tm models/other_models/modelsmodel12_train.pt -vm models/other_models/modelsmodel12_val.pt -mon modelsmodel14

or 

python3 train.py -tdp dataset/labels/test_new1.csv -tdr dataset/Images/test1/ -vdr dataset/Images/test2/ -vdp dataset/labels/test_new2.csv -tm models/other_models/modelsmodel12_train.pt -vm models/other_models/modelsmodel12_val.pt -mon 'modelsmodel14'

where:-
**dataset/labels/test_new1.csv** --> is the path to your train csv file that contains image_id and target
**dataset/Images/test1/** is the folder that contains your train images

**dataset/labels/test_new2.csv** --> is the path to your validation csv file that contains image_id and target
**dataset/Images/test2/** is the folder that contains your validation images

**models/other_models/modelsmodel12_train.pt** --> is the path to the previous train model checkpoint if available
**models/other_models/modelsmodel12_val.pt** --> is the path to the previous validation model checkpoint if available

*TODO* Write on how to use the test.py script
to run the program on a new testset you can use this code
python3 test.py -i dataset/labels/test_new2.csv -r dataset/Images/test2/ -vm models/other_models/modelsmodel12_train.pt

where:- 
**dataset/labels/test_new2.csv** --> is the path to your test csv file that contains image_id and target
**dataset/Images/test2/** --> is the folder that contains your test images
**models/other_models/modelsmodel12_train.pt** --> is the path to the model checkpoint that you want to test

*TODO* Write on the model.py script
The model.py script, is the script that contains the code that creates(or download and edit) the resnet50 architecture.

## Documentation
to see other arguments that can be passed to the train scrip run this command
python3 train.py -h

to see other arguments that can be passed to the test scrip run this command
python3 test.py -h

to see other arguments that can be passed to the process_data script run this command
python3 process_data.py -h


### Results for test datasets
| dataset            |    accuracy(%)| f1 score(%) |   precision(%) |   sensitvityi(%)  |loss     |
|--------------------|---------------|-------------|----------------|-------------------|---------|
|hold-out valset     |  0.7819       | 0.8302      | 0.9070         | 0.7819            |  0.6062 |
|hold-out testset    |  0.7744       | 0.8262      | 0.9093         | 0.77              |  0.6286 |
|indian dataset      |  0.8133       | 0.8310      | 0.8551         | 0.8133            |  0.5053 |
||||||



| dataset            |  specitivity(%) |
|--------------------|-----------------|
|hold-out valset     |     0.5295      |
|hold-out testset    |     0.5372      |
|indian dataset      |     0.7344      |
||||||


### Sizes of Datasets used
| dataset                               |      size     |
|---------------------------------------|---------------|
|diebetic-retinopathy-detection dataset |  35126        |
|messidor dataset                       |  1748         |
|indian dataset                         |  520          |


### Class size of datasets 
| dataset                   |   class 0 |   class 1   |  class 2   |
|---------------------------|-----------|-------------|------------|
|hold-out train set         |  21431    |   2173      |  5888      |
|hold-out validation set    |  2592     |   311       |  783       |  
|hold-out test set          |  2543     |   353       |  788       |
|indian test set            |  168      |   25        |  323       |

__Note__:- Due to the fact that some images were missing about 10 of them, the file name for those images was dropped

### Results on dataset splitting
| dataset            |      size     |
|--------------------|---------------|
|train set           |  29494        |
|test set            |  3686         |
|validation set      |  3688         |
||||||
