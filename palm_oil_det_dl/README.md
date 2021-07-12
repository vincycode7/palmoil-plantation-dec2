| Details                   |                        |
|---------------------------|------------------------|
| Programming Language:     |  Python 3.**8**        |
| pytorch version:          |  >=1.9.0+cu111         |
| pytorch lightning version |  >=0.8.5               |
| torchvision               |  0.10.0+cu111          |
| Models Required:          |pretrained resnet50 model|
| OS                        | Ubuntu 20              |



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

## If you prefer to go through miniconda use this (Miniconda starts here)
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

## If you prefer to go through miniconda use this (Miniconda ends here)


(17) if you have a gpu and you have linux run `sudo apt install nvidia-cuda-toolkit` to install package to enable gpu
(18) if experience timeout while downloading you can try apt-fast by following the steps below

    step 1-- > sudo apt-get install axel

    step 2 --> sudo add-apt-repository ppa:apt-fast/stable

    step 3 --> sudo apt-get update

    step 4 --> sudo apt-get -y install apt-fast

    step 5 --> sudo nano /etc/apt-fast.conf

    step 6 --> uncomment MIRROR in the apt-fast.conf file

    step 7 --> sudo apt-fast install nvidia-cuda-toolkit -y

    step 8 --> nvcc -V (in terminal to confirm cuda, if available run step 9)

    step 9 --> pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

###### Break out
