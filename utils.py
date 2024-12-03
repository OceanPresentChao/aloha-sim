import cv2
import numpy as np
import time
import h5py


def check_render():
    from mujoco.egl import egl_ext as EGL

    print("EGL.eglQueryDevicesEXT():", len(EGL.eglQueryDevicesEXT()))


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        # print(video[0])
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f"Saved video to: {video_path}")
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f"Saved video to: {video_path}")


def save_episode(episode, action, dataset_path):
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

    action                  (14,)         'float64'
    """
    assert len(episode) > 1

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    if len(action) > 0:
        assert len(action) == len(episode) - 1

    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/action": [],
    }
    camera_names = episode[0]["images"].keys()
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    data_dict["/action"] = action
    for obs in episode:
        data_dict["/observations/qpos"].append(obs["qpos"])
        data_dict["/observations/qvel"].append(obs["qvel"])
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                obs["images"][cam_name]
            )

    # HDF5
    assert str(dataset_path).endswith(".hdf5")

    t0 = time.time()
    with h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            _ = image.create_dataset(
                cam_name,
                (len(episode), 480, 640, 3),
                dtype="uint8",
                chunks=(1, 480, 640, 3),
            )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset("qpos", (len(episode), 14))
        _ = obs.create_dataset("qvel", (len(episode), 14))
        _ = obs.create_dataset("effort", (len(episode), 14))
        _ = root.create_dataset("action", (len(episode), 14))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f"Saving: {time.time() - t0:.1f} secs")

    return True
