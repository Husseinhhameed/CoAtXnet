def qlog(q):
    q = q / np.linalg.norm(q)
    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r = np.arctan2(sinhalftheta, coshalftheta)
    if sinhalftheta > 1e-6:
        qlog = r * q[1:] / sinhalftheta
    else:
        qlog = np.zeros(3)
    return qlog

def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]  # Translation components

    # Align and process rotation
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # Constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # Normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    return poses_out

def load_and_process_poses(data_dir, seqs, train=True, real=False, vo_lib='orbslam'):
    ps = {}
    vo_stats = {}
    all_poses = []
    for seq in seqs:
        seq_dir = os.path.join(data_dir, seq)
        p_filenames = [n for n in os.listdir(seq_dir) if n.find('pose') >= 0]
        if real:
            pose_file = os.path.join(data_dir, '{:s}_poses'.format(vo_lib), seq)
            pss = np.loadtxt(pose_file)
            frame_idx = pss[:, 0].astype(int)
            if vo_lib == 'libviso2':
                frame_idx -= 1
            ps[seq] = pss[:, 1:13]
            vo_stats_filename = os.path.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
            with open(vo_stats_filename, 'rb') as f:
                vo_stats[seq] = pickle.load(f)
        else:
            frame_idx = np.array(range(len(p_filenames)), dtype=int)
            pss = [np.loadtxt(os.path.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:12] for i in frame_idx if os.path.exists(os.path.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i)))]
            ps[seq] = np.asarray(pss)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        all_poses.append(ps[seq])

    all_poses = np.vstack(all_poses)
    pose_stats_filename = os.path.join(data_dir, 'pose_stats.txt')
    if train and not real:
        mean_t = np.mean(all_poses[:, [3, 7, 11]], axis=0)
        std_t = np.std(all_poses[:, [3, 7, 11]], axis=0)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
    else:
        mean_t, std_t = np.loadtxt(pose_stats_filename)

    # Process and normalize poses
    processed_poses = []
    for seq in seqs:
        pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                            align_s=vo_stats[seq]['s'])
        processed_poses.append(pss)

    return np.vstack(processed_poses)

# Define the transformation for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset class
class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.seqs = [seq for seq in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, seq))]
        self.samples = self._load_samples()
        self.processed_poses = self._load_processed_poses()

        # Debugging prints
        print(f"Number of samples: {len(self.samples)}")
        print(f"Number of processed poses: {self.processed_poses.shape[0]}")

        # Ensure consistency between samples and processed poses
        min_length = min(len(self.samples), self.processed_poses.shape[0])
        self.samples = self.samples[:min_length]
        self.processed_poses = self.processed_poses[:min_length]

    def _load_samples(self):
        samples = []
        for seq_folder in self.seqs:
            seq_path = os.path.join(self.root_dir, seq_folder)
            color_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.color.png')])
            depth_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.depth.png')])
            pose_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.pose.txt')])

            for color_file, depth_file, pose_file in zip(color_files, depth_files, pose_files):
                samples.append((os.path.join(seq_path, color_file),
                                os.path.join(seq_path, depth_file),
                                os.path.join(seq_path, pose_file)))
        return samples

    def _load_processed_poses(self):
        return load_and_process_poses(self.root_dir, self.seqs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.processed_poses):
            raise IndexError(f"Index {idx} out of bounds for processed poses of size {len(self.processed_poses)}")

        color_path, depth_path, pose_path = self.samples[idx]

        color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        pose_matrix = self.processed_poses[idx]

        if self.transform:
            color_image = self.transform(color_image)
            depth_image = (depth_image / depth_image.max() * 255).astype(np.uint8)
            depth_image = self.transform(depth_image)

        return color_image, depth_image, pose_matrix
