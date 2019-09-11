SHELL = bash
PC=python3
DEF_FILE= ./designs/aes/aes_Nangate45.def
POW_FILE= ./designs/aes/aes_Nangate45.pwr.rpt
LEF_FILE= ./platforms/nangate45/Nangate45.lef
CONGEST_RPT= ./designs/aes/aes_Nangate45.congest.rpt

TERM_SHELL= $(shell echo "$$0")
COMMAND =  $(shell source install.sh)
.PHONY: work templates 
.SILENT: build all

clean: work
	rm -rf ./work

work:  
	mkdir -p work
	mkdir -p templates
	mkdir -p work/parallel_runs

maps:
	mkdir -p input/current_maps
	$(PC) ./scripts/create_training_set.py

templates: 
	$(PC) ./src/create_template.py
	$(PC) ./src/eliminate_templates.py

data:
	$(PC) ./scripts/run_batch_iterative.py

training: 
	$(PC) ./src/cnn_train.py

parse_inputs:
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ${CONGEST_RPT}

predict:
	$(PC) ./src/cnn_inference.py

clean_release:
	$(PC) ./scripts/clean_release.py

install:
	$(PC) -m venv PDN-opt
ifeq ($(TERM_SHELL), bash)
	( \
		source PDN-opt/bin/activate; \
		pip3 install -r requirements.txt --no-cache-dir; \
		bash -c "source PDN-opt/bin/activate"; \
	)
else
	( \
		source PDN-opt/bin/activate.csh; \
		pip3 install -r requirements.txt --no-cache-dir; \
		tcsh -c "source PDN-opt/bin/activate.csh"; \
	)
endif

build:
	echo "****************************************************************"
	echo "****** Running scripts within PDN-opt virtual environment ******"
	echo "****************************************************************"
	echo "****************************************************************"
	echo "***** Extracting training sets and existing CNN checkpoints ****"
	echo "****************************************************************"
	$(PC) ./scripts/build.py
	echo "****************************************************************"
	echo "************************ Build completed ***********************"
	echo "****************************************************************"

all:
	rm -rf ./work
	mkdir -p work
	mkdir -p templates
	mkdir -p work/parallel_runs
	mkdir -p input/current_maps
	echo "****************************************************************"
	echo "************* Creating the defined templates *******************"
	echo "****************************************************************"
	$(PC) ./src/create_template.py
	$(PC) ./src/eliminate_templates.py
	echo "****************************************************************"
	echo "************* Creating the maps for SA *************************"
	echo "****************************************************************"
	$(PC) ./scripts/create_training_set.py
	echo "****************************************************************"
	echo "*** Running simulated annealing for training data collection ***"
	echo "****************************************************************"
	$(PC) ./scripts/run_batch_iterative.py
	echo "****************************************************************"
	echo "***************** Simulated annealing completed ****************"
	echo "****************************************************************"
	echo "****************************************************************"
	echo "********* Beginning CNN training with the golden data **********"
	echo "****************************************************************"
	$(PC) ./src/cnn_train.py
	echo "****************************************************************"
	echo "************* Creating the testcase current map ****************"
	echo "****************************************************************"
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE}
	echo "****************************************************************"
	echo "***************** Using CNN to synthesize PDN ******************"
	echo "****************************************************************"
	$(PC) ./src/cnn_inference.py
	echo "****************************************************************"
	echo "*** CNN-based PDN synthesized and stored in template_map.txt ***"
	echo "****************************************************************"
	echo "****************************************************************"
	echo "************* Running IR drop solver on the PDN ****************"
	echo "****************************************************************"
	$(PC) ./src/IR_map_generator.py

train:
	rm -rf ./work
	mkdir -p work
	mkdir -p templates
	mkdir -p work/parallel_runs
	mkdir -p input/current_maps
	$(PC) ./scripts/create_training_set.py
	$(PC) ./src/create_template.py
	$(PC) ./src/eliminate_templates.py
	$(PC) ./scripts/run_batch_iterative.py
	$(PC) ./src/cnn_train.py

inference:
	mkdir -p work
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ${CONGEST_RPT}
	$(PC) ./src/cnn_inference.py
	$(PC) ./src/IR_map_generator.py

get_ir:
	mkdir -p work
	$(PC) ./src/current_map_generator.py ${DEF_FILE} ${LEF_FILE} ${POW_FILE} ${CONGEST_RPT}
	$(PC) ./src/IR_map_generator.py

test:
	pytest
