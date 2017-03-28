-- Part 1.3 view.sql
--
-- Submitted by: Sari Nusier
--


-- add your CREATE VIEW statement here



CREATE VIEW `futureEventsView` AS
	SELECT * FROM `event` WHERE TIMESTAMP(`date`,`time`) > now()
WITH CHECK OPTION;


-- add your statement to test rejection
INSERT INTO `futureEventsView` VALUES (
 NULL,
 3,
 'Social',
 'Test futureEventsView',
 '2015-01-01',
 '11:11:00',
 'London'
);
