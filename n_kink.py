# ---------------------------- imports and settings ------------------
import os 
import numpy as np


# ---------------------------- class definition and methods ------------------
"""
METHODS
method1:
    read Z1+SP.dat and extract kink coordinates in every frame
    resort the coordinates of kinks according to x coordinate
    count number of kinks in every domain
method2:
    output information in a txt file:
    0-0.2   0.2-0.4   0.4-0.6   0.6-0.8   0.8-1
                    frame 1 information
                    frame 2 information
                            ...
                    
BASIC DATA STRUCTURES 
frame_information { (self-specified scale in the value dictionary can be 0.1...)
frame1: {0.2: a,
         0.4: b,
         0.6: c,
         0.6: d,
         0.8: e,
         1.0: f},
frame2: {0.2: a,
         0.4: b,
         0.6: c,
         0.6: d,
         0.8: e,
         1.0: f},
         ...,}
"""
class Kink:
    """compute number of kinks and separate at different domians along stretch direction"""
    def __init__(self, path, data_file, z1sp_file, n_chains):
        self.path = path
        self.data_file = data_file
        self.z1sp_file = z1sp_file
        self.n_chains = n_chains


    def extract_kinks(self):
        """compute number of kinks and separate with division domain"""
        data_file = os.path.join(self.path, self.data_file)
        kink_file = os.path.join(self.path, self.z1sp_file)
        # compute coordinate of box during simulation
        with open(data_file) as data:
            for line in data:
                stripped = line.strip()
                if stripped.endswith("xhi"):
                    xlo = float(stripped.split()[0])
                    xhi = float(stripped.split()[1])
                    x_c = (xlo+xhi) / 2
                if stripped.endswith("yhi"):
                    ylo = float(stripped.split()[0])
                    yhi = float(stripped.split()[1])
                    y_c = (ylo+yhi) / 2
                if stripped.endswith("zhi"):
                    zlo = float(stripped.split()[0])
                    zhi = float(stripped.split()[1])
                    z_c = (zlo+zhi) / 2

        # extract kink coordinates from all frames
        in_frame = False
        frame_count = 0
        frame_kink_coords = {}
        len_x = len_y = len_z = None
        xlo = ylo = zlo = None
        with open(kink_file) as z1:
            for line in z1:
                stripped = line.strip()
                if stripped == str(self.n_chains):
                    in_frame = True
                    frame_count += 1
                    frame_kink_coords[frame_count] = []
                    continue
                if not in_frame:
                    continue

                if len(stripped.split()) == 3:
                    len_x = float(stripped.split()[0])
                    len_y = float(stripped.split()[1])
                    len_z = float(stripped.split()[2])
                    xlo = x_c - len_x/2
                    xhi = x_c + len_x/2
                    ylo = y_c - len_y/2
                    yhi = y_c + len_y/2
                    zlo = z_c - len_z/2
                    zhi = z_c + len_z/2
                elif len(stripped.split()) == 7:
                    kink_x = float(stripped.split()[0])
                    kink_y = float(stripped.split()[1])
                    kink_z = float(stripped.split()[2])
                    if len_x is not None:
                        if kink_x > xhi:
                            kink_x = kink_x - len_x
                        elif kink_x < xlo:
                            kink_x = kink_x + len_x
                        else:
                            kink_x = kink_x
                        if kink_y > yhi:
                            kink_y = kink_y - len_y
                        elif kink_y < ylo:
                            kink_y = kink_y + len_y
                        else:
                            kink_y = kink_y
                        if kink_z > zhi:
                            kink_z = kink_z - len_z
                        elif kink_z < zlo:
                            kink_z = kink_z + len_z
                        else:
                            kink_z = kink_z
                        kink_xs = (kink_x-xlo) / len_x
                        kink_ys = (kink_y-ylo) / len_y
                        kink_zs = (kink_z-zlo) / len_z
                        frame_kink_coords[frame_count].append((kink_xs, kink_ys, kink_zs))
        for frame in frame_kink_coords:
            frame_kink_coords[frame].sort(key=lambda coord: coord[0])
        return frame_kink_coords
    

    def output_scaled_kinks(self, n_division, output=None):
        """output number of kinks"""
        # count number of kinks in every specified domain
        frame_kink_coords = self.extract_kinks()
        frame_kink_domains = {}
        scales = []
        for i in range(1, n_division+1):
            scales.append(i/n_division)

        for frame, scaled_coords in frame_kink_coords.items():
            frame_kink_domains[frame] = []
            i = 0
            n = len(scaled_coords)
            for scale in scales:
                while i < n and scaled_coords[i][0] <= scale:
                    i += 1
                frame_kink_domains[frame].append(i)

            abs_counts = [frame_kink_domains[frame][0]]
            for j in range(1, len(frame_kink_domains[frame])):
                abs_counts.append(frame_kink_domains[frame][j] - frame_kink_domains[frame][j-1])
            frame_kink_domains[frame] = abs_counts

        # output information in a txt file
        if output != None:
            output_file = os.path.join(self.path, output)
            with open(output_file, "w") as kink_info:
                for counts in frame_kink_domains.values():
                    line = " ".join(map(str, counts))
                    kink_info.write(line + "\n")
        return frame_kink_domains


# ---------------------------- execution tests ------------------
if __name__ == "__main__":
    path = "/Users/qingshiwang/Research/Python/Entanglement"
    data_file = "tensile_x.data"
    z1sp_file = "Z1+SP.dat"
    n_chains = 100
    kink_data = Kink(
        path=path, data_file=data_file, z1sp_file=z1sp_file, n_chains=n_chains
    )
    frame_kink_coords = kink_data.extract_kinks()
    frame_kink_domain = kink_data.output_scaled_kinks(n_division=10, output="kink.txt")
    #print(frame_kink_domain[1])





        
        
