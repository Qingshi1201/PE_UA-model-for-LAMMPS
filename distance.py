# ---------------------------- imports and settings ------------------
import os
import numpy as np



# ---------------------------- class definition and methods ------------------
"""
METHODS
method1:
    read Z1+SP.dat and compute average distance of kink-pair in a simulation
    average distance of different parallel simulations
method2:
    output averaged kink-pair distance frame with standard deviation

BASIC DATA STRUCTURES
kink_pairs {
chain1: {
        chain2: [(x1,y1,z1), (x2,y2,z2), ...],
        chain5: [(x1,y1,z1)]},
chain2: {
        chain1: [(x1,y1,z1), (x2,y2,z2), ...],
        chain6: [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)...],},
    ...   
}
kink_vectors = []
for chain in kink_pairs.keys():
    for kink_chain, kink_positions in kink_pairs.values():
        for position in kink_positions:
            idx = kink_positions.index(position)
            vector = position - kink_pairs[kink_chain][chain][idx]
    if chain == kink_chain:
    continue
    kink_vectors.append(vector)
"""
class KinkDistance:
    """compute average distance of kink-pair ends frame by frame"""
    def __init__(self, paths, z1_file, n_para, n_chains):
        self.paths = paths
        self.z1_file = z1_file
        self.n_chains = n_chains
        self.n_para = n_para


    def extract_frames(self, z1_path):
        """compute average distance of entanglements"""
        # extract all frame information from Z1+SP.dat
        current_frame = []
        frames = []
        with open(z1_path) as z1:
            for line in z1:
                stripped = line.strip()
                if stripped == str(self.n_chains):
                    if current_frame:
                        frames.append(current_frame)
                    current_frame = [stripped]
                else:
                    current_frame.append(stripped)
            if current_frame:
                frames.append(current_frame)
        return frames


    def compute_single_distance(self, z1_path):
        """within every frame extract kink coordinates of every chain"""
        frames = self.extract_frames(z1_path)
        frame_distances = []

        # extract kink pairs from different frames
        for frame_id, frame in enumerate(frames):
            chain_id = 0
            kinks = {}
            i = 2 # iterate over all lines but header of frames
            while i < len(frame):
                parts = frame[i].split()
                if len(parts) == 1:
                    chain_id += 1
                    chain_nkink = int(parts[0])
                    i += 1
                    for _ in range(chain_nkink):
                        info = frame[i].split()
                        if len(info) == 7:
                            x, y, z = map(float, info[:3])
                            kinked_chain = int(info[5])
                            if kinked_chain != chain_id:
                                key = (chain_id, kinked_chain)
                                kinks.setdefault(key, []).append(np.array([x, y, z]))
                        i += 1
                else:
                    i += 1

            # compute distances, average and std of kink-pairs
            distances = []
            visited = set()

            for (chain_1, chain_2), pos_1 in kinks.items():
                if (chain_2, chain_1) not in kinks:
                    continue
                pair = tuple(sorted((chain_1, chain_2)))
                if pair in visited:
                    continue
                pos_2 = kinks[(chain_2, chain_1)]
                for p1 in pos_1:
                    dmin = min(np.linalg.norm(p1-p2) for p2 in pos_2)
                    distances.append(dmin)

                visited.add(pair)

            frame_distances.append(distances)

        return frame_distances
    

    def compute_average_distance(self):
        """compute average distance from parallel simulations"""
        all_parallel = []

        for path in self.paths:
            z1_path = os.path.join(path, self.z1_file)
            all_parallel.append(self.compute_single_distance(z1_path))

        n_frames = len(all_parallel[0])
        dist_avg = []
        dist_std = []

        for f in range(n_frames):
            merged = []
            for p in range(self.n_para):
                merged.extend(all_parallel[p][f])

            if merged:
                dist_avg.append(np.mean(merged))
                dist_std.append(np.std(merged))
            else:
                dist_avg.append(0)
                dist_std.append(0)
        return dist_avg, dist_std
    
    
    def out_averge_distances(self, output_path, output_file):
        """write out average kink distances per frame in txt file"""
        os.makedirs(output_path, exist_ok=True)
        avg, std = self.compute_average_distance()

        output_file = os.path.join(output_path, output_file)
        data = np.column_stack((np.arange(len(avg)), avg, std))

        np.savetxt(
            output_file,
            data,
            fmt="%6d %15.6f %15.6f",
            header="frame_id avg_kink_distance std_kink_distance"
        )
        print(f"Saved averaged kink distances to {output_file}")



# ---------------------------- use and test ------------------
if __name__ == "__main__":
    base_path = "/Volumes/HardDevice/Winter26/Data/Tensile/formal6/z1data/amorphous/400K/1e8/"
    paths = []
    for i in range(1, 11):
        path = f"{base_path}/p{i}"
        paths.append(path)

    z1_file = "Z1+SP.dat"
    n_chains = 100
    n_para = 10
    output_path = base_path
    output_file = "kink_dist_avg.txt"

    kink_distance = KinkDistance(
        paths, z1_file, n_para, n_chains
    )
    kink_distance.out_averge_distances(output_path, output_file)