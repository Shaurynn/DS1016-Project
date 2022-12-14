# Data analysis
- DS1016-Project
- Description: Using Deep Learning to Detect Alzheimer's Disease
- Data Source: Alzheimer’s Disease Neuroimaging Initiative (https://adni.loni.usc.edu/)
- Type of analysis:

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for DS1016-Project in github.com/{group}. If your project is not set please add it:

Create a new project on github.com/{group}/DS1016-Project
Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "DS1016-Project"
git remote add origin git@github.com:{group}/DS1016-Project.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
DS1016-Project-run
```

# Install

Go to `https://github.com/{group}/DS1016-Project` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/DS1016-Project.git
cd DS1016-Project
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
DS1016-Project-run
```
