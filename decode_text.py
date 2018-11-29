#!/usr/bin/env python
"""
Script to convert binary format to text-like format
for DRS4 evaluation boards.
http://www.psi.ch/drs/evaluation-board

Jennifer Studer and Adrian Kulmburg
(studerje@student.ethz.ch and adrianku@student.ethz.ch)
Done on 09-11-2018 based on work by:
- Jonas Rembser (rembserj@phys.ethz.ch), 2016-04-15
- Gregor Kasieczka, ETHZ, 2014-01-15
- based on decode.C by Dmitry Hits
Note that only the end of the program was changed here,
and that the data contained in the header of the input
files is not saved in this version of decode.py.




"""

from sys import argv, exit
from numpy import array, uint32, cumsum, roll, zeros, float32, arange, shape,argmax, absolute,argmin,zeros_like
from struct import unpack
import matplotlib.pyplot as plt
import pandas as pd
import os as os
########################################
# Prepare Input
########################################

if not len(argv) == 2:
    print("Wrong number of arguments!")
    print("Usage: python decode.py filename.dat")
    print("Exiting...")
    exit()

maxtime=0
maxcounter=0
tempchannel_v=[]
valid_events=[[],[],[],[],[],[],[],[]]
input_filename = argv[1]
if not input_filename[-4:] == '.dat':
    print("Wrong arguments!")
    print("Usage: python decode.py filename.dat")
    print("Exiting...")
    exit()
input_filename = input_filename[:-4] # Just take the name without the extension

f = open( input_filename + '.dat', "rb")

# Prepare Output directory
if not os.path.exists(input_filename+"_Data"):
    os.makedirs(input_filename+"_Data")

########################################
# Actual Work
########################################

"""
Read in effective time width bins in ns and calculate the relative time from
the effective time bins. This is just a rough calculation before the correction 
done further down in the code

The script also gets channel number information from this section to create
the appropriate number of tree branches.

NOTE: the channels 0-3 are for the first board and channels 4-7 are for the second board
"""
# To hold to the total number of channels and boards
n_ch = 0
n_boards = 0

# Empty lists for containing the variables connected to the tree branches
channels_t = []
channels_v = []
channels_vlist =[]
channels_tlist =[]
# List of numpy arrays to store the time bin information
timebins = []
average_channel_v=[]

"""
This loop extracts time information for each DRS4 cell
"""
while True:
    header = f.read(4)
    # For skipping the initial time header
    if header == b"TIME":
        continue
    elif header.startswith(b"C"):
        n_ch = n_ch + 1
        # Create variables ...
        channels_t.append(zeros(1024, dtype=float32))
        channels_v.append(zeros(1024, dtype=float32))

        # Write timebins to numpy array
        timebins.append(array(unpack('f'*1024, f.read(4*1024))))

    # Increment the number of boards when seeing a new serial number
    # and store the serial numbers in the board serial numbers vector
    elif header.startswith(b"B#"):
        n_boards = n_boards + 1

    # End the loop if header is not CXX or a serial number
    elif header == b"EHDR":
        break

"""
# This is the main loop One iteration corresponds to reading one channel every
# few channels a new event can start We know that this if the case if we see
# "EHDR" instead of "C00x" (x=1..4) If we have a new event: Fill the tree, reset
# the branches, increment event counter The binary format is described in:
# http://www.psi.ch/drs/DocumentationEN/manual_rev40.pdf (page 24)
# What happens when multiple boards are daisychained: after the C004 voltages of
# the first board, there is the serial number of the next board before it starts
# again with C001.
"""

validcounter=0
current_board = 0
tcell = 0 # current trigger cell
t_00 = 0 # time in first cell in first channel for alignment
is_new_event = True
eventslist=[]
info_string = "Reading in events measured with {0} channels on {1} board(s)..."
print info_string.format(n_ch, n_boards)
valid_event=[[],[],[],[],[],[],[],[]]
time_valid_event=[[],[],[],[],[],[],[],[]]
testmin=[]

# This file writes all the date in different txt-like files (the extension
# .txt is not written, but you can still open them with any text editor).
# Ideally, we'd like to erase older files, in case you have to reuse this
# program. Thus, for each channel that is treated, if it is about to save
# the data for the first time in the corresponding file, we erase the
# previous file and store the number of that channel in the following
# variable:
written_channels = []
# That way, we will erase old data, but not the new data that we want to
# write in it.

# Also, in order to track how many events were already analyzed, use this variable.
current_event_n = 1


while True:
    
    var=0
    delta_t=[]

    # Start of Event
    if is_new_event:
        is_new_event = False

        # Set the timestamp, where the milliseconds need to be converted to
        # nanoseconds to fit the function arguments
        dt_list = unpack("H"*8, f.read(16))

        # Fluff the serial number and read in trigger cell
        fluff = f.read(4)
        tcell = unpack('H', f.read(4)[2:])[0]
        # Reset current board number
        current_board = 0
        continue

    # Read the header, this is either
    #  EHDR -> finish event
    #  C00x -> read the data
    #  ""   -> end of file
    header = f.read(4)

    # Handle next board
    if header.startswith(b"B#"):
        current_board = current_board + 1
        tcell = unpack(b'H', f.read(4)[2:])[0]
        continue

    # End of Event
    elif header == b"EHDR":
        # Fill previous event
        is_new_event = True
        print "Done with event number ", current_event_n
        current_event_n += 1

    # Read and store data
    elif header.startswith(b"C"):
        # the voltage info is 1024 floats with 2-byte precision
        chn_i = int(header.decode('ascii')[-1]) + current_board * 4
        scaler = unpack('I', f.read(4))
        voltage_ints = unpack(b'H'*1024, f.read(2*1024))

        """
        Calculate precise timing using the time bins and trigger cell
        see p. 24 of the DRS4 manual for the explanation
        the following lines sum up the times of all cells starting from the trigger cell
        to the i_th cell and select only even members, because the amplitude of the adjacent cells are averaged.
        The width of the bins 1024-2047 is identical to the bins 0-1023, that is why the arrays are simply extended
        before performing the cumsum operation
        """
        timebins_full = list(roll(timebins[chn_i-1], -tcell))+list(roll(timebins[chn_i-1], -tcell))
        t = cumsum(timebins_full)[::2]
        # time of first cell for correction, find the time of the first cell for each channel,
        # because only these cells are aligned in time
        t_0 = t[(1024-tcell)%1024]
        if chn_i % 4 == 1:
            t_00 = t_0
        # Align all channels with the first channel
        t = t - (t_0 - t_00) # correction
        # TODO: it is a bit unclear how to do the correction with
        # TODO: multiple boards, so the boards are just corrected independently for now
        # TODO: find the alignment of the boards by sending the same signal to both boards
        """
        The following lists of numpy arrays can be plotted or used in the further analysis.
        
        NOTE: the channels 0-3 are for the first board and channels 4-7 are for the second board
        """
        if chn_i in written_channels:
            # In case we have already written in the file corresponding to chn_i, we can just append
            # the data...
            channels_v_file = open(input_filename+"_Data/"+input_filename+"_chn{}_v".format(chn_i), "a")
            channels_t_file = open(input_filename+"_Data/"+input_filename+"_chn{}_t".format(chn_i), "a")
        else:
            # ...and in case we haven't, we just erase the (potentially already existing) old file...
            channels_v_file = open(input_filename+"_Data/"+input_filename+"_chn{}_v".format(chn_i), "w")
            channels_t_file = open(input_filename+"_Data/"+input_filename+"_chn{}_t".format(chn_i), "w")
            # ... and add chn_i to the list of channels we have already treated at least once.
            written_channels.append(chn_i)
        for i, x in enumerate(voltage_ints):
            channels_v[chn_i-1][i] = ((x / 65535.) - 0.5)
            channels_t[chn_i-1][i] = t[i]
        
            channels_v_file.write(str((x / 65535.) - 0.5))
            channels_v_file.write(" ") # This is just to separate the different entries
            channels_t_file.write(str(t[i]))
            channels_t_file.write(" ")
        channels_v_file.write("\n") # In order to separate the different events
        channels_t_file.write("\n")
        channels_v_file.close()
        channels_t_file.close()




        triggertimelist=[]
        
        
    if (header == ""):
            break
print "Done with event number ", current_event_n
print "Your data is ready."
f.close()

