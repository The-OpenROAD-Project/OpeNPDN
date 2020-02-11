SHELL = bash
PC=python3
## For multiple LEF files including cell LEF include with a space in quotes
LEF_FILE= "./platforms/nangate45/NangateOpenCellLibrary.mod.lef"
DEF_FILE= "./test/aes/aes.def"
POW_FILE= "./test/aes/aes_pwr.rpt"
CONGEST_RPT= ""
CHECKPOINT_DIR = "./checkpoints"


# Location of OpenDB python wrapper. Do not change if OpenDB built successful
ODB_LOC = "./build/modules/OpenDB/src/swig/python/opendbpy.py"
PDN_CFG = "./input/PDN.cfg"
TOOL_CFG = "./input/tool_config.json"
TECH_SPEC = "./input/tech_spec.json"
CONGESTION_ENABLED = 0

TERM_SHELL= $(shell echo "$$0")
COMMAND =  $(shell source install.sh)
.PHONY: work templates data test
.SILENT: build all test


ifeq (${CONGESTION_ENABLED}, 0)
CONGESTION_COMMAND = "no_congestion"
CONGEST_RPT = "congestion_report_invalid"
else
CONGESTION_COMMAND = ""
endif

clean: work
	rm -rf ./work

work:  
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs 

settings:
	$(PC) ./src/T6_PSI_settings.py ${ODB_LOC} ${CHECKPOINT_DIR} ${LEF_FILE} ${PDN_CFG} ${TOOL_CFG} ${TECH_SPEC}

maps:
	mkdir -p input/current_maps &&\
	$(PC) ./scripts/create_training_set.py 

templates: 
	$(PC) ./src/T6_PSI_settings.py ${ODB_LOC} ${CHECKPOINT_DIR} ${LEF_FILE} ${PDN_CFG} ${TOOL_CFG} ${TECH_SPEC} &&\
	mkdir -p templates
	$(PC) ./src/create_template.py 

parse_inputs:
	$(PC) ./src/current_map_generator.py ${POW_FILE} ${LEF_FILE}  ${DEF_FILE} ${CONGESTION_COMMAND} ${CONGEST_RPT}

predict:
	$(PC) ./src/cnn_inference.py ${CONGESTION_COMMAND}

release:
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

build_OpeNPDN:
	echo "****************************************************************" &&\
	echo "****** Running scripts within PDN-opt virtual environment ******" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "***** Extracting training sets and existing CNN checkpoints ****" &&\
	echo "****************************************************************" &&\
	$(PC) ./scripts/build.py &&\
	echo "****************************************************************" &&\
	echo "************************ Build completed ***********************" &&\
	echo "****************************************************************"

all:
	rm -rf ./work ./templates &&\
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs &&\
	mkdir -p input/current_maps &&\
	mkdir -p checkpoints &&\
	echo "****************************************************************" &&\
	echo "************* Creating the defined templates *******************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/T6_PSI_settings.py ${ODB_LOC} ${CHECKPOINT_DIR} ${LEF_FILE}  ${PDN_CFG} ${TOOL_CFG} ${TECH_SPEC} &&\
	$(PC) ./src/create_template.py &&\
	echo "****************************************************************" &&\
	echo "**** Creating the maps for training data generation ************" &&\
	echo "****************************************************************" &&\
	$(PC) ./scripts/create_training_set.py &&\
	echo "****************************************************************" &&\
	echo "******** Running golden data generation  ***********************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/generate_training_data_iterative.py ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "***************** Data generation completed ********************" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "********* Beginning CNN training with the golden data **********" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/cnn_train.py ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "************* Creating the testcase current map ****************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/current_map_generator.py ${POW_FILE} ${LEF_FILE}  ${DEF_FILE} ${CONGESTION_COMMAND} ${CONGEST_RPT} &&\
	echo "****************************************************************" &&\
	echo "***************** Using CNN to synthesize PDN ******************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/cnn_inference.py ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "*** CNN-based PDN synthesized and stored in template_map.txt ***" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "************* Running IR drop solver on the PDN ****************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/IR_map_generator.py

training_set:
	rm -rf ./work &&\
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs &&\
	mkdir -p input/current_maps &&\
	mkdir -p checkpoints &&\
	$(PC) ./src/T6_PSI_settings.py ${ODB_LOC} ${CHECKPOINT_DIR} ${LEF_FILE} ${PDN_CFG} ${TOOL_CFG} ${TECH_SPEC}  &&\
	$(PC) ./scripts/create_training_set.py 
	
data:
	$(PC) ./src/T6_PSI_settings.py ${ODB_LOC} ${CHECKPOINT_DIR} ${LEF_FILE} ${PDN_CFG} ${TOOL_CFG} ${TECH_SPEC}  &&\
	$(PC) ./src/create_template.py &&\
	$(PC) ./src/generate_training_data_iterative.py ${CONGESTION_COMMAND}

data_and_train:
	rm -rf ./work &&\
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs &&\
	mkdir -p input/current_maps &&\
	mkdir -p checkpoints &&\
	$(PC) ./src/T6_PSI_settings.py ${ODB_LOC} ${CHECKPOINT_DIR} ${LEF_FILE} ${PDN_CFG} ${TOOL_CFG} ${TECH_SPEC}  &&\
	$(PC) ./scripts/create_training_set.py &&\
	$(PC) ./src/create_template.py &&\
	$(PC) ./src/generate_training_data_iterative.py ${CONGESTION_COMMAND} &&\
	$(PC) ./src/cnn_train.py ${CONGESTION_COMMAND}

train:
	$(PC) ./src/cnn_train.py ${CONGESTION_COMMAND}

inference:
	mkdir -p work &&\
	$(PC) ./src/current_map_generator.py ${POW_FILE} ${LEF_FILE}  ${DEF_FILE} ${CONGESTION_COMMAND} ${CONGEST_RPT} &&\
	$(PC) ./src/cnn_inference.py ${CONGESTION_COMMAND} && \
	$(PC) ./src/IR_map_generator.py

get_ir:
	mkdir -p work &&\
	$(PC) ./src/current_map_generator.py ${POW_FILE} ${LEF_FILE}  ${DEF_FILE} ${CONGESTION_COMMAND} ${CONGEST_RPT} &&\
	$(PC) ./src/IR_map_generator.py

TEST_LEF_FILE= "./platforms/nangate45/NangateOpenCellLibrary.mod.lef"
TEST_DEF_FILE= "./test/aes/aes.def"
TEST_POW_FILE= "./test/aes/aes_pwr.rpt"
TEST_PDN_CFG = "./test/PDN.cfg"
TEST_TOOL_CFG = "./test/tool_config.json"
TEST_TECH_SPEC = "./test/tech_spec.json"

test:
	rm -f ./test/aes.log &&\
	touch ./test/aes.log &&\
	rm -rf ./work ./templates &&\
	mkdir -p work &&\
	mkdir -p templates &&\
	mkdir -p work/parallel_runs &&\
	mkdir -p input/current_maps &&\
	mkdir -p checkpoints &&\
	echo "****************************************************************" &&\
	echo "************* Creating the defined templates *******************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/T6_PSI_settings.py ${ODB_LOC} ${CHECKPOINT_DIR} ${TEST_LEF_FILE} ${TEST_PDN_CFG} ${TEST_TOOL_CFG} ${TEST_TECH_SPEC} | tee -a ./test/aes.log &&\
	$(PC) ./src/create_template.py | tee -a ./test/aes.log &&\
	echo "****************************************************************" &&\
	echo "**** Creating the maps for training data generation ************" &&\
	echo "****************************************************************" &&\
	$(PC) ./scripts/create_training_set.py | tee -a ./test/aes.log &&\
	echo "****************************************************************" &&\
	echo "******** Running golden data generation  ***********************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/generate_training_data_iterative.py ${CONGESTION_COMMAND} &&\
	echo "****************************************************************" &&\
	echo "***************** Data generation completed ********************" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "********* Beginning CNN training with the golden data **********" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/cnn_train.py ${CONGESTION_COMMAND}   &&\
	echo "****************************************************************" &&\
	echo "************* Creating the testcase current map ****************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/current_map_generator.py ${TEST_POW_FILE} ${TEST_LEF_FILE}  ${TEST_DEF_FILE} ${CONGESTION_COMMAND} ${CONGEST_RPT} | tee -a ./test/aes.log  &&\
	echo "****************************************************************" &&\
	echo "***************** Using CNN to synthesize PDN ******************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/cnn_inference.py ${CONGESTION_COMMAND} | tee -a ./test/aes.log  &&\
	echo "****************************************************************" &&\
	echo "*** CNN-based PDN synthesized and stored in template_map.txt ***" &&\
	echo "****************************************************************" &&\
	echo "****************************************************************" &&\
	echo "************* Running IR drop solver on the PDN ****************" &&\
	echo "****************************************************************" &&\
	$(PC) ./src/IR_map_generator.py | tee -a ./test/aes.log 
	diff ./test/aes.log ./test/aes.ok
	cmp -s ./test/aes.log ./test/aes.ok; \
	RETVAL=$$?; \
	if [ $$RETVAL -eq 0 ]; then \
	        echo "TEST PASS"; \
	else \
	        echo "TEST FAIL"; \
	fi
