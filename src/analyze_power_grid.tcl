#BSD 3-Clause License
#
#Copyright (c) 2019, The Regents of the University of Minnesota
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

sta::define_cmd_args "analyze_power_grid" {
    [-OPDN_DIR OPDN_DIR]\

# Put helper functions in a separate namespace so they are not visible
# to users in the global namespace.
namespace eval openpdn {
    variable db
    variable opendbpy
    variable OPDN_DIR
    variable TEMPALTE_CFG
    
    #This file contains procedures that are used for PDN generation
    proc file_exists {filename} {
	return [expr {([file exists $filename]) && ([file size $filename] > 0)}]
    }

    proc init {opendb_db} {
        variable db
        variable OPDN_DIR
        variable opendbpy
        variable TEMPALTE_CFG
       
        set opendbpy [file normalize $::env(OPDN_OPENDBPY)]
        if {![file_exists $opendbpy]} {
               sta::sta_error "File for opendbpy.py \"$opendbpy\" does not exist, or exists but empty"
        }
        set TEMPALTE_CFG [file normalize $::env(TEMPLATE_PGA_CFG)]
        if {![file_exists $TEMPALTE_CFG]} {
               sta::sta_error "File for template_pga.cfg \"$TEMPALTE_CFG\" does not exist, or exists but empty"
        }
        set OPDN_DIR [file normalize $::env(OPDN_SRC)]
        if {![file isdirectory ${OPDN_DIR}]} {
               sta::sta_error "Directory location for OpeNPDN \"$OPDN_DIR\" does not exist, or not defined"
        }
        set  db $opendb_db 
    }


    proc openpdn { verbose } {
        variable db
        variable opendbpy
        variable checkpoints
        variable OPDN_DIR
        variable TEMPALTE_CFG

        file mkdir ${OPDN_DIR}/work
        write_db "${OPDN_DIR}/work/PDN.db"
        
	    if {$verbose} {
	        puts "Running per instance power report"
	    }
        set openpdn_congestion_enable "no_congestion"
        set WD [pwd]
        
        cd ${OPDN_DIR}
        
        foreach x [get_cells *] {
        	set y [get_property $x full_name]
        	report_power -instance $y -digits 10 >> ./work/power_instance.rpt
        	}
        
        set OPDN_MODE "INFERENCE"
        
	    if {$verbose} {
	        puts "Creating required templates"
	    }
        exec python3 src/T6_PSI_settings.py "${opendbpy}" "" "${TEMPALTE_CFG}" "${OPDN_MODE}"
        
        file mkdir templates
        exec python3 src/create_template.py
        
	    if {$verbose} {
	        puts "Generating IR map and report"
	    }
        exec python3 src/current_map_generator.py work/power_instance.rpt $openpdn_congestion_enable
        exec python3 src/IR_map_generator.py
        
        puts "Results stored in ${OPDN_DIR}/output"
        file delete -force -- ${OPDN_DIR}/work
        
        cd ${WD}
    }

}

proc analyze_power_grid { args } {
   sta::parse_key_args "analyze_power_grid" args \
   keys {} \
   flags {-verbose}

  # if [info exists flags(-help)] {
  #     puts "Usage: openpdn -OPDN_DIR <OpeNPDN path> -opendbpy <opendbpy.py path> "
  #     return 0
  # }
    set verbose [info exists flags(-verbose)]

    sta::check_argc_eq0 "analyze_power_grid" $args
    set db [ord::get_db]
    openpdn::init  $db
    openpdn::openpdn  $verbose
}
