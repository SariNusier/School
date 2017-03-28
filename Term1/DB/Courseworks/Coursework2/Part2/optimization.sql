-- Part 2.2 optimization.sql
--
-- Submitted by: Sari Nusier
--


--  Write your SQL statements here

ALTER TABLE image ADD COLUMN crc32 int unsigned;
ALTER TABLE image ADD INDEX (`crc32`);
UPDATE image SET crc32 = CRC32(image);


-- An and should be added to find the image wanted when multiple results are returned (same checksums)
SELECT image_no WHERE crc32 = '25965481';
