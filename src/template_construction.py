#BSD 3-Clause License
#
#Copyright (c) 2019, The Regents of the University of Minnesota
#
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:10:27 2018
Template definitions and building the G matrix
@author: Vidya A Chhabria
Defines the template class that is used for to store
all the attributes of the template. Defines how to constuct that base G is
generic amd cmake all the necessary conditions.
"""
import math
import time
import fractions as frac
import numpy as np
import scipy.sparse as sparse_mat


class template_def():
    """ Class that defines the templates using the infromation obtained from the
    JSON file.
    Attributes:
        n_layers: number of layers in the PDN template
        pitches: vector of pitches of each layer in the template
        widths: vector of widths of metal stripefor each layer in the template
        rho_s: vector of sheet resistances for every layer in the template
        via_res: vector of via resitances for all the vias used in the template
        dirs: A vector of 0's and 1's representing the direction of every layer
              used in the template
        xpitch: x-direction pitch of the grid on which the PDN is laid out
        ypitch: y-direction pitch of the grid on which the PDN is laid out
        num_x: Number of nodes in the grid in x direction
        num_y: Number of nodes in the grid in the y direction
        start: Array of locations for the begiinning of every template
        init_offset: Offset initialization for determinning the start of the
                     first stripe and ensuring the last stripe does not go
                     beyond the area
        G: Conductance matrix for every template
    """

    def __init__(self, n_layers, pitches, widths, rho_s, via_res, dirs,
                 chip_length, chip_width):
        """Initializes the template_def class attributes"""
        self.n_layers = n_layers
        self.pitches = pitches
        self.widths = widths
        self.rho_s = rho_s
        self.via_res = via_res
        self.dirs = dirs
        self.xpitch = self.get_x_grid_pitch(self.pitches, self.dirs)
        self.ypitch = self.get_y_grid_pitch(self.pitches, self.dirs)
        self.xpitch,self.y_pitch = self.refine_grid_pitch(self.xpitch,
                                    self.ypitch,self.dirs[0])
        self.num_x = 1 + math.floor(chip_width / self.xpitch)
        self.num_y = 1 + math.floor(chip_length / self.ypitch)
        self.start = self.create_G_start()
        self.init_offset = self.create_init_offset()
        self.G = []
        self.offset = np.zeros(((self.pitches.shape[0]), 1), dtype=int)

    def refine_grid_pitch(self,xpitch,ypitch,M1_dir):
        if M1_dir == 0:
            #m1 is horizontal and has a y pitch
            ref_y_pitch = ypitch
            ref_x_pitch = xpitch
            while ref_x_pitch >  2*ref_y_pitch :
                ref_x_pitch = ref_x_pitch/2
        else:
            #m1 is vertical and has a x pitch
            ref_y_pitch = ypitch
            ref_x_pitch = xpitch
            while ref_y_pitch >  2*ref_x_pitch :
                ref_y_pitch = ref_y_pitch/2
        return ref_x_pitch,ref_y_pitch


    def get_y_grid_pitch(self, pitches, dirs):
        """ This function finds the pitch of the grid in the y direction. THis
        is the grid on which the PDN lies on. The grid pitch is determined by
        taking the GCD of the pitch of every layer in the template which has
        been defined based on the DRC rules.
        Args:
            pitches: Array of the pitch of every layer in a template
            dirs: Array of 0's and 1's represeting the direction of every metal
                 layer in the PDN
        Returns
            gcd_val: grid pitch in y direction after taking gcd of pitches
        """
        # dirs should be 0 horizontal wires pitches determine the y pitch
        pitch_arr = pitches[dirs == 0] * 1e9
        pitch_arr = np.round(pitch_arr)
        pitch_arr = np.array(pitch_arr)
        gcd_val = np.amin(pitch_arr)
        for pitch in pitch_arr:
            # Find the gcd of the pitches to determine grid pitch
            gcd_val = frac.gcd(gcd_val, pitch)
        # Check if templates are stitchable
        if (gcd_val < (pitches[0] * 1e9) and dirs[0] == 0):
            print(
                "############################################################",
                "##################################################")
            print(
                "GCD VALUE ERROR. Templates may not be stichable.",
                "Check the pitch values in the template_definition.xlsx file")
            print(
                "#############################################################",
                "#################################################")
        # other directions only error out if it goes too low
        elif gcd_val < (pitches[0] * 1e9) / 4:
            print(
                "############################################################",
                "##################################################")
            print(
                "GCD VALUE ERROR. Templates may not be stichable.",
                "Check the pitch values in the template_definition.xlsx file")
            print(
                "#############################################################",
                "#################################################")
        return np.round(gcd_val) * 1e-9

    def get_x_grid_pitch(self, pitches, dirs):
        """ This function finds the pitch of the grid in the y direction. THis
        is the grid on which the PDN lies on. The grid pitch is determined by
        taking the GCD of the pitch of every layer in the template which has
        been defined based on the DRC rules.
        Args:
            pitches: Array of the pitch of every layer in a template
            dirs: Array of 0's and 1's represeting the direction of every metal
                 layer in the PDN
        Returns
            gcd_val: grid pitch in x direction after taking gcd of pitches
        """
        # dirs should be 1 vertical wires pitches determine the x pitch
        pitch_arr = pitches[dirs == 1] * 1e9
        pitch_arr = np.round(pitch_arr)
        gcd_val = np.amin(pitch_arr)
        pitch_arr = np.array(pitch_arr)
        for pitch in pitch_arr:
            # Find the gcd of the pitches to determine grid pitch
            gcd_val = frac.gcd(gcd_val, pitch)
        # Check if templates are stitchable
        # all nodes on lower layer must be on a rail so that we can calulate IR
        if (gcd_val < (pitches[0] * 1e9) and dirs[0] == 1):
            print(
                "#################################################",
                "#############################################################")
            print(
                "GCD VALUE ERROR. Templates may not be stichable.",
                "Check the pitch values in the template_definition.xlsx file")
            print(
                "#################################################",
                "#############################################################")
        # other directions only error out if it goes too low
        elif gcd_val < (pitches[0] * 1e9) / 4:
            print(
                "#################################################",
                "#############################################################")
            print("GCD VALUE ERROR. Templates may not be stichable."
                  "Check the pitch values in the template_definition.xlsx file")
            print(
                "#################################################",
                "#############################################################")
        return np.round(gcd_val) * 1e-9

    def create_init_offset(self):
        """ This function sets the offset of the metal stripe to ensure the
        current source at the edge is connected to a stripe and not floating.
        Also determines the beginning of the metal stripe
        Args:
            pitches: Array of the pitch of every layer in a template
            dirs: Array of 0's and 1's represeting the direction of every metal
                 layer in the PDN
        Returns
            buf: An array of offsets for every layer
        """
        buf = np.zeros(((self.pitches.shape[0]), 1), dtype=int)
        for i, direc in enumerate(self.dirs):
            if direc == 0:    # horizontal use y pitch
                num_y_metal = round(self.pitches[i] / self.ypitch)
                buf[i] = int((self.num_y % num_y_metal) / 2)
            else:
                num_x_metal = round(self.pitches[i] / self.xpitch)
                buf[i] = int((self.num_x % num_x_metal) / 2)
        return buf

    def check_index(self, in_i, in_j, nx_i, nx_j, grid_pitch, metal_pitch,
                    offset):
        """ This function checks if the current index is at the edge of chip
        This determines the connections to every resistor.
        Args:
            in_i: Current index in the x direction
            in_j: Current index in the y direction
            nx_i: Next index in the x direction
            ny_j: Next index in the y direction
            grid_pitch: Grid pitch in the current direction for which the G
            matrix is begin construted
            metal_pitch: Pitch of the metal for which the current G matrix is
            being built
        Returns
            res: Flag variable to tell whether the current node is at the
            boundary of the chip. If res == 0, then it is a boundary. Otherwise,
            it is not a boundary
        """
        res = 1
        # check and eliminate boundaries
        if nx_i < 0:
            res = 0
        elif nx_i >= self.num_x:
            res = 0
        elif nx_j < 0:
            res = 0
        elif nx_j >= self.num_y:
            res = 0
        else:    # not a boundary
            if abs(nx_j - in_j) > 0:    # vertical cnnection
                if (
                    (np.round(((in_i - offset) * grid_pitch) * 1e9) %
                     np.round(metal_pitch * 1e9)) < 1
                        and (in_i - offset) * grid_pitch >= 0
                ):    # current index is at a node that has a veritcal connection
                    res = 1
                else:
                    res = 0
            elif abs(nx_i - in_i) > 0:    # horizontal cnnection
                if (
                    (np.round(((in_j - offset) * grid_pitch) * 1e9) %
                     np.round(metal_pitch * 1e9)) < 1
                        and (in_j - offset) * grid_pitch >= 0
                ):    # current index is at a node that has a horizontal connection
                    res = 1
                else:
                    res = 0
            else:    # vertical connection always exits
                res = 1
        return res

    def convert_index(self, i, j):
        """ This function converts two indices which representa a location in a
        2D array into a unique number. Ordered by row major
        Args:
            i: Current index in the x direction
            j: Current index in the y direction
        Returns
            index_temp: Unique integer value representing the position of every
            element in a 2D matrix
        """
        # If it is at the boundary
        if ((i < 0) or (i >= self.num_x) or (j < 0) or (j >= self.num_y)):
            index_temp = 0
        else:
            index_temp = self.num_x * (j) + i
        return index_temp

    def conductivity(self, i, j, width, length, rho):
        """ This function provides the conductance value that is used to
        populate the G matrix. Basically performs:conductance = width/(rho*len)
        Args:
            i: Current index in the x direction
            j: Current index in the y direction
            width: Width of the metal stripe forwhich the conductance value is
            being estimate
            length: Distance to the next node
            rho: Sheet resistance of the metal
        Returns
            index_temp: Unique integer value representing the position of every
            element in a 2D matrix
        """
        # conductivity is along the directin of the metal
        if i < 0:
            cond = 0
        elif i >= self.num_x:
            cond = 0
        elif j < 0:
            cond = 0
        elif j >= self.num_y:
            cond = 0
        else:
            cond = width / (rho * length)
        return cond

    def create_G_single_layer(self, width, rho, metal_pitch, dirs, offset):
        """ This function provides creates the portion of the G matrix for a
        single layer anf single region.
        Args:
            width: Width of the metal stripe of the concerned layer
            rho: Sheet resistance of the layer in consideration
            metal_pitch: pitch of the stripes for the current layer
            dirs: Direction of the metal layer
            offset: Distance to the edge of the chip
        Returns
            G: Conductance matrix for the single region and single layer
        """
        G = sparse_mat.dok_matrix(
            (self.num_x * self.num_y, self.num_x * self.num_y))
        for j in range(self.num_y):
            for i in range(self.num_x):
                # Set the pitch of the grid on whihc the metal stripe needs to
                # be laid out on
                if dirs == 0:    # horizontal
                    np_i = i + 1
                    np_j = j
                    nm_i = i - 1
                    nm_j = j
                    grid_pitch = self.ypitch
                    grid_pitch_orth = self.xpitch
                else:    # vertical
                    np_i = i
                    np_j = j + 1
                    nm_i = i
                    nm_j = j - 1
                    grid_pitch = self.xpitch
                    grid_pitch_orth = self.ypitch
                # Check the location of the current index if it is a boundary or
                # not and fill in the conuddctance value accordingly
                if (template_def.check_index(self, i, j, np_i, np_j, grid_pitch,
                                             metal_pitch, offset)):
                    G[template_def.convert_index(self, i, j), template_def.convert_index(self, np_i, np_j)] = \
                        -1 * template_def.conductivity(self, np_i, np_j, width, grid_pitch_orth, rho)

                    G[template_def.convert_index(self, i, j),
                      template_def.convert_index(self, i, j)] = G[
                          template_def.convert_index(self, i, j),
                          template_def.convert_index(
                              self, i, j)] + template_def.conductivity(
                                  self, np_i, np_j, width, grid_pitch_orth, rho)
                if (template_def.check_index(self, i, j, nm_i, nm_j, grid_pitch,
                                             metal_pitch, offset)):
                    G[template_def.convert_index(self, i, j), template_def.convert_index(self, nm_i, nm_j)] = \
                        -1 * template_def.conductivity(self, nm_i, nm_j, width, grid_pitch_orth, rho)

                    G[template_def.convert_index(self, i, j),
                      template_def.convert_index(self, i, j)] = G[
                          template_def.convert_index(self, i, j),
                          template_def.convert_index(
                              self, i, j)] + template_def.conductivity(
                                  self, nm_i, nm_j, width, grid_pitch_orth, rho)
        return G

    def create_G(self):
        """ This function  creates the portion of the G matrix for a
        single region and all layers and updates the object.
        """
        # For all layers in the PDN create the single layer G matrix and create
        # the G matrix for the region for all layers by using connect_vas
        # function
        for i in range(self.n_layers):
            G_single = template_def.create_G_single_layer(
                self, self.widths[i], self.rho_s[i], self.pitches[i],
                self.dirs[i], self.offset[i])
            if i == 0:
                self.G = G_single
                self.start[i] = 0
            else:
                self.G = sparse_mat.block_diag((self.G, G_single), format='dok')
                self.start[i] = self.start[i - 1] + self.num_x * self.num_y
        # Add a via at every intersection between two metal layers
        for i in range(self.num_x):
            for j in range(self.num_y):
                for n in range(self.n_layers - 1):
                    for k in range(n + 1, self.n_layers):
                        if not self.dirs[k] == self.dirs[n]:
                            break
                    self.G = template_def.connect_vias(
                        self, self.G, i, j, self.pitches[k], self.pitches[n],
                        self.via_res[n], self.start[k], self.start[n],
                        self.offset[k], self.offset[n], self.dirs[n])

    def create_G_start(self):
        """Create an array which contains an index of the begining of the
        portion of the G matrix for every layer.
        Returns:
            start: An n_layer X 1 array which has the index location for the
            beginning of every layer for a given region in G
        """
        start = np.zeros(((self.pitches.shape[0]), 1), dtype=int)
        for i in range(self.n_layers):
            if i == 0:
                start[i] = 0
            else:
                start[i] = start[i - 1] + self.num_x * self.num_y
        return start

    def connect_vias(self, G, i, j, pitch1, pitch2, via_res, start1, start2,
                     offset1, offset2, dirs):
        """Creates the connection between two adjacent metal layers in the PDN
        using vias a t every point of intersection between them.
        Args:
            G: Conductance matrix for all the layers without the connections
            beween them for all regions
            i: Current index
            j: Next index
            pitch1: Grid pitch of the first of the two connecting layers
            pitch2: Grid pitch of the second of the two connecting layers
            via_res: Resistane of the via between the two connecting layers
            start1: Starting index in the G matrix for the first layer
            start2: Starting index in the G matrix for the second layer
            offset1: Offset of the first layer
            offset2:Offset of the second layer
        Returns:
            G: The conductance matrix the with the two layers connected with
        """
        # Find the direction of the connecting layers
        if dirs == 0:    #horizontal
            y_start = start2
            x_start = start1
            pitch_x = pitch1
            pitch_y = pitch2
            offset_y = offset2
            offset_x = offset1
        else:    # verticle
            y_start = start1
            x_start = start2
            pitch_x = pitch2
            pitch_y = pitch1
            offset_y = offset1
            offset_x = offset2
        # Find the point of intersectio between the two layers and add the value
        # of conductance for the via
        # Have to handle arrays of cias appropriately as of now just scaling the
        # conductance
        if (((np.round((
            (i - offset_x) * self.xpitch) * 1e9) % np.round(pitch_x * 1e9)) < 1)
                and ((np.round(((j - offset_y) * self.ypitch) * 1e9) %
                      np.round(pitch_y * 1e9)) < 1)
                and (i - offset_x) * self.xpitch >= 0
                and (j - offset_y) * self.ypitch >= 0):
            G[x_start + template_def.convert_index(self, i, j), y_start +
              template_def.convert_index(self, i, j)] = -1 / via_res
            G[y_start + template_def.convert_index(self, i, j), x_start +
              template_def.convert_index(self, i, j)] = -1 / via_res
            G[x_start + template_def.convert_index(self, i, j), x_start +
              template_def.convert_index(self, i, j)] = G[
                  x_start + template_def.convert_index(self, i, j), x_start +
                  template_def.convert_index(self, i, j)] + 1 / via_res
            G[y_start + template_def.convert_index(self, i, j), y_start +
              template_def.convert_index(self, i, j)] = G[
                  y_start + template_def.convert_index(self, i, j), y_start +
                  template_def.convert_index(self, i, j)] + 1 / via_res
        return G


# main for testing if G is created properly
if __name__ == '__main__':
    c_length = 1e-4
    c_width = 1e-4

    pitch_values = np.array(
        [0.27e-6, 0.27e-6, 8.64e-6, 5.67e-6, 2.97e-6, 2.97e-6, 2.97e-6])
    width_values = np.array(
        [0.018e-6, 0.018e-6, 0.432e-6, 0.576e-6, 0.576e-6, 1.92e-6, 1.98e-6])
    rho_values = np.array([1.22, 1.22, 0.77, 0.5, 0.5, 0.36, 0.36])
    via_resistance = np.array([17.2, 46.2, 11.8, 8.2, 8.2, 6.3])
    directions = np.array([1, 0, 1, 0, 1, 0, 1])
    t0 = time.time()
    template_obj = template_def(7, pitch_values, width_values, rho_values,
                                via_resistance, directions, c_length, c_width)
    t1 = time.time()
    total_time = t1 - t0
    G = sparse_mat.dok_matrix.tocoo(template_obj.G)
    f = open("ouputG.txt", "w+")
    for row, col, value in zip(G.row, G.col, G.data):
        f.write("({0}, {1}) {2} \n".format(row, col, value))
    f.close()
