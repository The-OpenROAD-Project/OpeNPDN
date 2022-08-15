# OpeNPDN: Neural-network-based framework for Power Delivery Networks (PDN) Synthesis
[![Standard](https://img.shields.io/badge/python-3.6-blue)](https://commons.wikimedia.org/wiki/File:Blue_Python_3.7_Shield_Badge.svg)
[![Download](https://img.shields.io/badge/Download-here-red)](https://github.com/The-OpenROAD-Project/OpeNPDN/archive/master.zip)
[![Version](https://img.shields.io/badge/version-1.0-green)](https://github.com/The-OpenROAD-Project/OpeNPDN/tree/master)
[![AskMe](https://img.shields.io/badge/ask-me-yellow)](https://github.com/The-OpenROAD-Project/OpeNPDN/issues)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Machine learning-based on-chip power delivery network (PDN) synthesis at the placement stage.  The synthesis is based on a set of predefined, technology-specific set of templates. These templates are defined across multiple layers and vary in their metal utilizations in the intermediate layers. Dense templates are good for power integrity but bad for congestion. The problem of optimized PDN synthesis is converted into one of finding a template in every region on the tiling of a chip as shown in the figure below:

<img align = "right" width="45%" src="doc/image.png">

This problem is solved as a classification problem using a convolution neural network (CNN). The computationally expensive cost of analyzing and optimizing the PDN is encapsulated into a one-time training step of the CNN. Using the trained CNN, for a specific PDK and region size, a correct-by-construction PDN can be quickly synthesized for any design as shown in the figure on the right:


## Machine Learning Flow for PDN Synthesis


- Inference and training:
  * Input definition:
    * PDN templates defined by a combination of numbers defined in [templates.csv](params/templates.csv) and [grid parameters](params/grid_params.json) file
    * Current maps: A 2D distribution of current across the chip specified on a per-region basis as shown in [current map](designs/aes/current_maps.csv). 
    * Congestion maps: A 2D distribution of congestion across the chip specified on a per-region basis in the same format as the current map.
    * Macro maps: A specification of lower left and upper right corners of all macros in the design as specified by [macros.txt](designs/bp_be/macro.txt)
    * Power bump distribution file: A specification of the x and y coordinates of the power bump location with its voltage source value [vsrc.txt](designs/bp_be/vsrc.txt)
    * The VDD value and IR drop limit as specified by [IR_params.json](params/IR_params.json)
  * Output definition:
    * Template map across the chip saved in a generated output file called output_templates.png

<img align = "right" width="50%" src="doc/flow.png">

The training data generation is a one-time step which involves running a simulated annealing based PDN optimization for multiple current maps.  This part of the flow needs to be run only **once** for a particular PDK and *region* size. Once the CNN has been trained for a given PDK, the *inference flow* can be run on any given design for the fixed *region* size. The region size is defined in the JSON file.  

## Getting Started
### Prerequisite
- python 3.6
- pip 18.1
- python3-venv

Additionally, please refer to [requirements.txt](requirements.txt) file in this repository. 
The packages in requirements.txt will be installed in a virtual environment during build.

### Install on a bare-metal machine

#### Clone repo and submodules
```
git clone --recursive https://github.com/The-OpenROAD-Project/OpeNPDN.git
```

#### Setup environment for OpeNPDN with bash shell
```
cd OpeNPDN
python3 -m venv openpdn
source openpdn/bin/activate
pip3 install -r requirements.txt
```

#### Running OpeNPDN

Training consists of five stages: 
1. Template elimination
2. Synthetic feature generation
3. Golden label generation using SA for synthetic and real circuit data
4. Pretraining CNN with synthetic data
5. Transfer learning CNN with real circuit data


The real circuit data must be specified in a subfolder in the [designs](designs/) directory
which contains four files per design corresponding to the four features -- currents, congestion, bump distributions,
and macro location specification as defined in an earlier section of the README.


Steps to run:

```
mkdir run run/checkpoint run/data run/TL_data
python3 src/template_elimination.py
python3 src/synthetic_data_generation.py
python3 src/CNN_synth_training.py
python3 src/TL_data_gen.py
python3 src/TL_CNN_training.py
```


Steps to run:

Inference requires the path of the design folder, the path of the final CNN
checkpoint as input. The design folder must contain all the features.

```
python3 src/inference.py <path to design folder with testcase> <path to CNN
checkpoint>
```


## LICENSE

The default template pitches, resistances, current maps and all data available on
this repository is for the Nangate45nm PDK.The Nangate45nm PDK is downloaded 
from https://projects.si2.org/openeda.si2.org/project/showfiles.php?group_id=63#503 

The rest of this repository is licensed under BSD 3-Clause License.

>BSD 3-Clause License
>
>Copyright (c) 2021, The Regents of the University of Minnesota
>
>All rights reserved.
>
>Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
>* Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
>* Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
>* Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
>AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
>IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
>DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
>FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
>DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
>SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
>CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
>OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
>OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
