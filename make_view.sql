drop view if exists data_points;
create view data_points_view as
WITH time_differences AS (
	SELECT
		id,
		timestamp,
		-- LAG(timestamp) OVER (ORDER BY timestamp) AS prev_timestamp,
		LAG(timestamp) OVER (ORDER BY id) AS prev_timestamp,
		timestamp - lag(timestamp) over (order by timestamp) as time_diff_nanos
	FROM images
),

group_boundaries AS (
	SELECT
		timestamp,
		CASE
			WHEN time_diff_nanos IS NULL OR time_diff_nanos > 166666667 THEN 1
			ELSE 0
		END AS is_new_group
	FROM time_differences
),

group_numbers AS (
	SELECT
		timestamp,
		SUM(is_new_group) OVER (ORDER BY timestamp) AS group_id
	FROM group_boundaries
)
	SELECT
		images.id as img_id,
		scans.id as scan_id,
		occupancy_grids.id as grid_id,
		group_numbers.group_id,
		group_numbers.timestamp,
		images.image,
		scans.ranges,
		poses.x,
		poses.y,
		poses.theta,
		occupancy_grids.data
	FROM
		images join
		scans
			on images.timestamp = scans.timestamp join
		occupancy_grids
			on occupancy_grids.timestamp = images.timestamp join
		group_numbers ON images.timestamp = group_numbers.timestamp join
		poses on poses.timestamp = group_numbers.timestamp;

create index scan_ts on scans(timestamp);
create index image_ts on images(timestamp);
create index occupancy_grid_ts on occupancy_grids(timestamp);
create index pose_ts on poses(timestamp);
