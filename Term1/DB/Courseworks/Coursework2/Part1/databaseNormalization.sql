-- Part 1.5 databaseNormalization.sql
--
-- Submitted by: Sari Nusier
--


--  Write your Database Normalization statements here

-- This table contains the status codes and their corresponding descriptions.
CREATE TABLE IF NOT EXISTS `crimeStatus` AS
	SELECT DISTINCT `status` AS 'status_code',
		`status_desc`
	FROM `crimes`;

ALTER TABLE `crimeStatus` ADD PRIMARY KEY(`status_code`);


-- This table contains the area names and codes.
CREATE TABLE IF NOT EXISTS `area` AS
	SELECT DISTINCT `area` AS 'area_code',
		`area_name`
	FROM `crimes`;

ALTER TABLE `area` ADD PRIMARY KEY(`area_code`);


-- This table contains the type of crimes (codes and descriptions)
CREATE TABLE IF NOT EXISTS `crimeType` AS
	SELECT DISTINCT `crime_no`,
		`crime_desc`
	FROM `crimes`;

DELETE FROM `crimeType` WHERE `crime_desc` = 'THEFT PLAIN - PETTY (UNDER $400)';
DELETE FROM `crimeType` WHERE `crime_no` = 813;
INSERT INTO `crimeType` VALUES(813, 'CHILD ENDANGERMENT/NEG. OR CHILD ANNOYING (17YRS & UNDER, DID NOT TOUCH VICTIM)');
DELETE FROM `crimeType` WHERE `crime_no` = 930;
INSERT INTO `crimeType` VALUES(930, 'CRIMINAL THREATS - NO WEAPON DISPLAYED OR THREATS, VERBAL/TERRORIST');

ALTER TABLE `crimeType` ADD PRIMARY KEY(`crime_no`);


-- The location is given by the road number (rd). This references the area,
-- which can be found in the table created above.
CREATE TABLE IF NOT EXISTS `location` AS
	SELECT DISTINCT `rd` AS 'rd_id',
		`area` AS `area_code`
	FROM `crimes`;

-- Primary and foreign key
ALTER TABLE `location`
ADD PRIMARY KEY(`rd_id`),
    CONSTRAINT `road_area` FOREIGN KEY(`area_code`) REFERENCES `area` (`area_code`);


-- This table contains an occurance of one crime.
CREATE TABLE IF NOT EXISTS `crime` AS
	SELECT `dr_no`,
		`date_reported`, `date_occ`, `time_occ`,
    `rd` AS `location_id`,
    `crime_no` AS `crime_type`,
    `status` AS `crime_status`,
    `image_no`
	FROM `crimes`;

ALTER TABLE `crime`
	ADD PRIMARY KEY(`dr_no`),
	ADD CONSTRAINT `location_fk` FOREIGN KEY(`location_id`) REFERENCES `location` (`rd_id`),
	ADD CONSTRAINT `type_fk` FOREIGN KEY(`crime_type`) REFERENCES `crimeType` (`crime_no`),
	ADD CONSTRAINT `status_fk` FOREIGN KEY(`crime_status`) REFERENCES `crimeStatus` (`status_code`),
	ADD CONSTRAINT `image_fk` FOREIGN KEY(`image_no`) REFERENCES `image` (`image_no`)
;
