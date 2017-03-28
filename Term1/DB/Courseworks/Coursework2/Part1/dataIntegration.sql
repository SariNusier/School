-- Part 1.4 dataIntegration.sql
--
-- Submitted by: Sari Nusier
--


--  Write your Data Integration statements here

CREATE TABLE IF NOT EXISTS `crimes` (
  `dr_no` INT NOT NULL,
  `date_reported` DATE NOT NULL,
  `date_occ` DATE NOT NULL,
  `time_occ` TIME NOT NULL,
  `area` INT NOT NULL,
  `area_name` VARCHAR(30) NOT NULL,
  `rd` INT NOT NULL,
  `crime_no` INT NOT NULL,
  `crime_desc` VARCHAR(100) NOT NULL,
  `status` VARCHAR(10) NOT NULL,
  `status_desc` VARCHAR(100) NOT NULL,
  `image_no` INT,
  PRIMARY KEY (`dr_no`),
  CONSTRAINT `crime_image`
    FOREIGN KEY (`image_no`)
    REFERENCES `image` (`image_no`)
);

INSERT INTO crimes
	SELECT * FROM crimes2013
	UNION
	SELECT * FROM crimes2014
	UNION
	SELECT * FROM crimes2015;
