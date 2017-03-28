-- Part 1.3 dataPreprocessing.sql
--
-- Submitted by: Sari Nusier
--


-- Write your Data Preprocessing statements here

-- We first move seconds to minutes and minutes to the hours positions,
-- for both crimes2013 and 2014 tables.
UPDATE crimes2013
SET time_occ = time_format(time_occ, '%i:%s:00');

UPDATE crimes2014
SET time_occ = time_format(time_occ, '%i:%s:00');

-- We set the type of time_occ column to TIME, in order to keep a uniform type
-- across all tables.
-- This is not needed as the conversion will happen when the merging of the tables
-- will be done in the next part.
ALTER TABLE crimes2015
MODIFY time_occ TIME;

UPDATE crimes2013
SET image_no = NULL WHERE image_no <= 0;

UPDATE crimes2014
SET image_no = NULL WHERE image_no <= 0;

UPDATE crimes2013
SET status = UPPER(status);

UPDATE crimes2013
SET date_reported = STR_TO_DATE(date_reported, '%D %M, %Y'),
date_occ = STR_TO_DATE(date_occ, '%D %M, %Y');

UPDATE crimes2014
SET date_occ = STR_TO_DATE(date_occ, '%Y-%M-%d'),
date_reported = STR_TO_DATE(date_reported, '%Y-%M-%d');

UPDATE crimes2015
SET date_occ = STR_TO_DATE(date_occ, '%m/%d/%Y'),
date_reported = STR_TO_DATE(date_reported, '%m/%d/%Y');
