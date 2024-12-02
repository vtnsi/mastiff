<img align="right" width="400" src="https://github.com/user-attachments/assets/1c4ec069-6b95-42c2-8cb3-9171dd4e1f1e">

# Welcome to the MASTIFF GYM Python Package!
The MASTIFF Gym is intended as a research environment for *adversarial* testing and evaluation (T&E) of radio frequency machine learning (RFML) models. MASTIFF Gym is an adversarial adaptation of the original [RFRL Gym](https://github.com/vtnsiSDD/rfrl-gym?tab=readme-ov-file)

## To install the codebase:

### Setting up within a virtualenv

Creating a development setup for mastiff makes use of a `virtualenv` and an extension for bash specifically `virtualenv-better-bash`

1. Install dependencies:

```bash
if [ $UID = 0 ]; then
apt install -y git automake autoconf apt-utils pkg-config \
    bc zip unzip gfortran libfftw3-dev dialog cmake libcairo2-dev libgirepository1.0-dev \
    python3 python3-dev python3-pip python3-apt python3-yaml python3-pygccxml
else
sudo apt install -y git automake autoconf apt-utils pkg-config \
    bc zip unzip gfortran libfftw3-dev dialog cmake libcairo2-dev libgirepository1.0-dev \
    python3 python3-dev python3-pip python3-apt python3-yaml python3-pygccxml
fi
```

Note: If spinning up in a fresh docker image, sudo is not required above

2. Install virtualenv-better-bash:

```bash
pip install --user virtualenv-better-bash
```

3. Clone repository:

```bash
git clone git@github.com:maymoonaht/mastiff.git
```

4. Setup virtualenv (either in the repo, or put the repo inside a virtualenv):

 - Setting up the virtualenv inside the repo
```bash
# setting up virtualenv inside the project
virtualenv --prompt mastiff --activators better_bash mastiff/.env
export VENV_ROOT="$(pwd)/mastiff/.env"
export DEV_ROOT="$(pwd)/mastiff"
mkdir ${VENV_ROOT}/src
cd mastiff
```
 - Setting up the repo inside the virtualenv
```bash
# setting up the project inside the virtualenv
mv mastiff mastiff-dev
virtualenv --activators better_bash mastiff
export VENV_ROOT="$(pwd)/mastiff"
mkdir ${VENV_ROOT}/src
mv mastiff-dev ${VENV_ROOT}/src/mastiff
cd ${VENV_ROOT}/src/mastiff
export DEV_ROOT="$(pwd)"
```


Note: using envrionment variables to help track better

5. Activate the virtual environment:

```bash
source ${VENV_ROOT}/bin/activate
```

6. Install mastiff:

```bash
pip install ${DEV_ROOT}
```

7. Install GNU Radio:

For simplicity, installing with `PyBOMBS` is the easiest approach (not apt).
This step can be skipped if gnuradio libraries are already reachable (calling `PyBOMBS` setup_env.sh)

```bash
pip install pybombs pybind11
# if installing gnuradio independent of mastiff
#     pybombs prefix init <path>/gnuradio
#     export GR_ROOT=<path>/gnuradio
# else
#     export GR_ROOT=${VENV_ROOT}
ln -s "${VENV_ROOT}/lib/python3.10/pybind11/include/pybind11" "${VENV_ROOT}/include/pybind11"
export GR_ROOT="${VENV_ROOT}"
pybombs prefix init ${GR_ROOT} -a mastiff
pybombs -p mastiff recipes add-defaults
pybombs -p mastiff config makewidth 8 ## install threads
if [ $UID = 0 ]; then
    pybombs -p mastiff config elevate_pre_args ''
fi
pybombs -p mastiff fetch uhd
sed -i "/^if(NOT DEFINED USRP_MPM_PYTHON_DIR)/i set(USRP_MPM_PYTHON_DIR \"${GR_ROOT}/lib/python3.10/site-packages\")" \
    ${GR_ROOT}/src/uhd/mpm/python/CMakeLists.txt
## update recipes
###### no longer needed
#cp "${DEV_ROOT}/recipes/gr-foo.lwr" "${GR_ROOT}/.pybombs/recipes/gr-recipes/"
#cp "${DEV_ROOT}/recipes/gr-ieee-80211.lwr" "${GR_ROOT}/.pybombs/recipes/gr-recipes/"
#cp "${DEV_ROOT}/recipes/gr-ieee-802154.lwr" "${GR_ROOT}/.pybombs/recipes/gr-etcetera/"
## install gnuradio (sudo will be asked for if installed)
pybombs -p mastiff -y -q install gnuradio
### extra that likely isn't needed
if ! [ "${GR_ROOT}" = "${VENV_ROOT}" ]; then
    cat "${GR_ROOT}/setup_env.sh" | head -n -4 >> "${VENV_ROOT}/bin/activate"
fi
###
pybombs -p mastiff -y -q install gr-foo gr-ieee-80211 gr-ieee-802154
grcc "${GR_ROOT}/src/gr-ieee-80211/examples/wifi_phy_hier.grc"
mv "wifi_phy_hier.py" \
   "${GR_ROOT}/lib/python3.10/site-packages/ieee802_11/"

if ! [ -d "${HOME}/.config/gnuradio/prefs" ]; then
    mkdir -p "${HOME}/.config/gnuradio/prefs"
fi
echo -n "gr::vmcircbuf_mmap_shm_open_factory" > ${HOME}/.config/gnuradio/prefs/vmcircbuf_default_factory
```

8. Install Liquid-DSP:

There's a slight tweak to Liquid-DSP while it's python integration is on-going.
For now this gets the job done.

```bash
cd "${VENV_ROOT}/src"
git clone --single-branch --branch dev/bind-dev https://github.com/SaikWolf/liquid-dsp.git
cd liquid-dsp
./bootstrap.sh
./configure --prefix="${VENV_ROOT}"
make -j8 install python
cd "${VENV_ROOT}/lib/python3.10/site-packages"
ln -s "${VENV_ROOT}/src/liquid-dsp/liquid.cpython-310-x86_64-linux-gnu.so"
```

Note: This might be overkill now, `pip install liquid-dsp` might be enough for what we use.

9. Download the best.pt and zigbee_best.pt files and place them in the /src/detection folder

[best.pt](https://drive.google.com/file/d/1m5ms4h-jir9ecX0x3nYvLRFcpINKCGmV/view?usp=sharing)

[best_zigbee.pt](https://drive.google.com/file/d/1UDSS7AKrrewF7V8gt_5AEkbkbcyzJh6c/view)

## To test installation of the codebase and the renderer:
```bash
python3 scripts/test_adversarial_skeleton.py
```

A terminal output should print out showing the observation space upon successful execution. 

## Example #1: Basic Scenario
This scenario tests a MUT using YOLO that can detect any signal of any bandwidth. The goal of the RL adversarial agent is to manipulate the environment through changing the power of the signals. Optimal results should show a maximum reward when power is minimized. The resulting MUT performance will show a decrease in the number of detected bounding boxes.

```bash
python3 examples/basic/train_adversarial_dqn.py -s basic_scenario.json -m adversarial -a mastiff_gym_dqn_basic -e 100
```
To change the scenario, use '-s'. To change the environment, use '-m'. To change the name of the trained agent model, use '-a'. To change the number of epochs to train over, use '-e'.

```bash
python3 examples/basic/test_adversarial_dqn.py -s basic_scenario.json -m adversarial -a mastiff_gym_dqn_basic
```
A GUI should pop up displaying first the results of the agent before learning and then the results of the agent after learning.

To change the scenario, use '-s'. To change the environment, use '-m'. To change the pre-trained agent model, use '-a'.

## Example #2: Intermediate Scenario
This scenario tests a MUT using YOLO that can detect only zigbee signals. The goal of the RL adversarial agent is to manipulate the environment through changing the power AND bandwidth of the signals. Optimal results should show a maximum reward when power is minimized and the bandwidth is not zigbee's bandwidth. The resulting MUT performance will show a decrease in the number of detected bounding boxes.

```bash
python3 examples/intermediate/train_adversarial_ppo.py -s intermediate_scenario.json -m adversarial -a mastiff_gym_ppo_intermediate -e 10
```
To change the scenario, use '-s'. To change the environment, use '-m'. To change the name of the trained agent model, use '-a'. To change the number of epochs to train over, use '-e'.

```bash
python3 examples/intermediate/test_adversarial_ppo.py -s intermediate_scenario.json -m adversarial -a mastiff_gym_ppo_intermediate
```
A GUI should pop up displaying first the results of the agent before learning and then the results of the agent after learning.

To change the scenario, use '-s'. To change the environment, use '-m'. To change the pre-trained agent model, use '-a'.

## How to reference:
To reference MASTIFF Gym: TBD

To reference original RFRL Gym:
```
@inproceedings{rfrlgym,
  Title = {{RFRL Gym: A Reinforcement Learning Testbed for Cognitive Radio Applications}},
  Author = {D. Rosen, I. Rochez, C. McIrvin, J. Lee, K. D’Alessandro, M. Wiecek, N. Hoang, R. Saffarini, S. Philips, V. Jones, W. Ivey, Z. Harris-Smart, Z. Harris-Smart, Z. Chin, A. Johnson, A. Jones, W. C. Headley},
  Booktitle = {{IEEE International Conference on Machine Learning and Applications (ICMLA)}},
  Year = {2023},
  Location = {Jacksonville, USA},
  Month = {December},
  Url = {}
```

# Data generation tools

More details can be found [here](DATAGEN.md)

# YOLOv7 adaptation for MASTIFF

A fork of the original [YOLOv7 repo](https://github.com/WongKinYiu/yolov7) was created [here](https://github.com/maymoonaht/yolov7-mastiff) to train on a dataset created using the above data generation tool.

