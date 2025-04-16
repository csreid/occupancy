from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
	# Declare the launch arguments
	node_file_path = LaunchConfiguration('node_file_path')
	poetry_project_dir = LaunchConfiguration('poetry_project_dir')
	poetry_bin_path = LaunchConfiguration('poetry_bin_path', default='/usr/local/bin/poetry')

	# Find the nav2 package
	nav2_bringup_dir = FindPackageShare('nav2_bringup')
	nav2_params_path = LaunchConfiguration('nav2_params_path', default='nav2_params.yml')

	return LaunchDescription([
		# Declare nav2 params path argument
		DeclareLaunchArgument(
			'nav2_params_path',
			default_value='nav2_params.yml',
			description='Path to the Nav2 parameters file'
		),

		# Include Nav2 launch
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource(
				PathJoinSubstitution([nav2_bringup_dir, 'launch', 'bringup_launch.py'])
			),
			launch_arguments={
				'use_sim_time': 'True',
				'slam': 'True',
				'params_file': nav2_params_path,
			}.items()
		),
		ExecuteProcess(
			cmd=['rviz2'],
			output='screen'
		),
	])
