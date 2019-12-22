set test_dir [file dirname [file normalize [info script]]]
set openroad_dir [file dirname [file dirname [file dirname $test_dir]]]
set OPDN_DIR [file join $test_dir "/OpenROAD/src/OpeNPDN"]
set opendbpy [file join $openroad_dir "/OpenROAD/build/src/OpenDB/src/swig/python/opendbpy.py"]
#puts $openroad_dir

#set OPDN_DIR "/home/sachin00/chhab011/tmp/OpenROAD/src/OpeNPDN/"
#set opendbpy "/home/sachin00/chhab011/OpenDB/build/src/swig/python/opendbpy.py"
#read_lef  ../platforms/nangate45/NangateOpenCellLibrary.mod.lef
#read_def aes/aes.def
#read_liberty ../platforms/nangate45/NangateOpenCellLibrary_typical.lib
#read_sdc aes/aes.sdc

read_lef  ../platforms/nangate45/NangateOpenCellLibrary.mod.lef
read_def aes/aes.def
read_liberty ../platforms/nangate45/NangateOpenCellLibrary_typical.lib
read_sdc aes/aes.sdc


set checkpoints "./OpeNPDN-Checkpoint-FreePDK45"

#set test_dir [file dirname [file normalize [info script]]]
#set openroad_dir [file dirname [file dirname [file dirname $test_dir]]]
#set OPDN_DIR [file join "src/OpeNPDN"]
#set opendbpy [file join "build/src/swig/python/opendbpy.py"
#puts $test_dir

#set OPDN_DIR "/home/sachin00/chhab011/OpenROAD/src/OpeNPDN/"
#set opendbpy "/home/sachin00/chhab011/OpenDB/build/src/swig/python/opendbpy.py"


run_openpdn  -OPDN_DIR ${OPDN_DIR} -opendbpy ${opendbpy} -checkpoints ${checkpoints} -verbose
exit 0
