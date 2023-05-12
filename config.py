
model = {
    "map_size": 40,#50,
    "min_len": 8,#3,  # minimal possible distance in meters
    "max_len": 15,#12,#15,  # maximal possible disance to go straight in meters
    "min_pos": 1,  # minimal possible 
    "max_pos": 38, #49 29,  # maximal possible position in meters
    "grid_size": 1,  # 1,  # size of the grid in meters
    "robot_radius": 0.1,  # 1,  # radius of the robot in meters
    "start": 1,  # 1,  # start x position
    "goal": 38,  # 1,  # start y position
}

files = {
    "logs_path": ".\\2023-01-28_logs\\",
    "model_path": ".\\2023-01-28_models\\",
    "img_path": ".\\2023-01-28_images\\", #".\\2022-10-18_images\\"
}