-- Part 1.2 dataLoading.sql
--
-- Submitted by: Sari Nusier
--


-- Part 1.2.1 Table Creation

--
-- In both cases we declare time_occ of type TIME

CREATE TABLE IF NOT EXISTS `crimes2013` (
  `dr_no` INT NOT NULL,
  `date_reported` VARCHAR(30) NOT NULL,
  `date_occ` VARCHAR(30) NOT NULL,
  `time_occ` TIME NOT NULL,
  `area` INT NOT NULL,
  `area_name` VARCHAR(30) NOT NULL,
  `rd` INT NOT NULL,
  `crime_no` INT NOT NULL,
  `crime_desc` VARCHAR(100) NOT NULL,
  `status` VARCHAR(10) NOT NULL,
  `status_desc` VARCHAR(100) NOT NULL,
  `image_no` INT,
  PRIMARY KEY (`dr_no`)
);

CREATE TABLE IF NOT EXISTS `crimes2014` (
  `dr_no` INT NOT NULL,
  `date_reported` VARCHAR(30) NOT NULL,
  `date_occ` VARCHAR(30) NOT NULL,
  `time_occ` TIME NOT NULL,
  `area` INT NOT NULL,
  `area_name` VARCHAR(30) NOT NULL,
  `rd` INT NOT NULL,
  `crime_no` INT NOT NULL,
  `crime_desc` VARCHAR(100) NOT NULL,
  `status` VARCHAR(10) NOT NULL,
  `status_desc` VARCHAR(100) NOT NULL,
  `image_no` INT,
  PRIMARY KEY (`dr_no`)
);


-- Part 1.2.1 Data Load
LOAD DATA LOCAL INFILE '/home/sari/Development/School/DB/Courseworks/Coursework2/Part1/data/crimes2013.txt'
INTO TABLE crimes2013
FIELDS ENCLOSED BY '\''
IGNORE 1 LINES
(@dr_no, @date_reported,  @date_occ, @time_occ ,@area ,@area_name ,@rd ,@crime_no ,@crime_desc ,@status ,@status_desc ,@image_no)
SET `crime_no` = @crime_no,
`rd`=@rd,
`area`=@area,
`dr_no`=@dr_no,
`area_name`=@area_name,
`date_reported`=@date_reported,
`crime_desc`=@crime_desc,
`status_desc`=@status_desc,
`image_no`=@image_no,
`date_occ`=@date_occ,
`status`=@status,
`time_occ`=@time_occ;

LOAD DATA LOCAL INFILE '/home/sari/Development/School/DB/Courseworks/Coursework2/Part1/data/crimes2014.csv'
INTO TABLE crimes2014
FIELDS TERMINATED BY ';' ENCLOSED BY '\''
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(@date_reported, @dr_no, @date_occ, @time_occ ,@area ,@area_name ,@rd ,@crime_no ,@crime_desc ,@status ,@status_desc ,@image_no)
SET `crime_no` = @crime_no,
`rd`=@rd,
`area`=@area,
`dr_no`=@dr_no,
`area_name`=@area_name,
`date_reported`=@date_reported,
`crime_desc`=@crime_desc,
`status_desc`=@status_desc,
`image_no`=@image_no,
`date_occ`=@date_occ,
`status`=@status,
`time_occ`=@time_occ;
