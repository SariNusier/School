-- Part 1.4 select.sql
--
-- Submitted by: Sari Nusier
--


-- add your SELECT statements here

-- 1.4.1 Employee List.
SELECT `name`, `surname`, `email` FROM `employee`;

-- 1.4.2 Oldest Status.
SELECT `text` FROM `status` ORDER BY `timestamp` ASC limit 1;

-- 1.4.3 Total Usage.
SELECT FORMAT(SUM(`total`), 0) AS `totalUsage`
FROM (
	SELECT COUNT(*) AS `total` FROM `status`
  UNION
  SELECT COUNT(*) AS `total` FROM `comment`
  UNION
  SELECT COUNT(*) AS `total` FROM `event`
) AS allTables;

-- 1.4.4 Event Report
-- ASSUMING THAT WE WANT THE MONTH OF FEBRUARY OF THE CURRENT YEAR!
SELECT `type`, FORMAT(COUNT(*), 0) AS `totalEvents`
FROM `event`
WHERE MONTH(`date`) = 2 AND YEAR(`date`) = YEAR(CURDATE())
GROUP BY `type`;

-- 1.4.5 Comment Report

SELECT
  CONCAT(`name`, ' ', `surname`) AS `fullName`,
  (SELECT
          COUNT(*)
      FROM
          `comment`
      WHERE
          `creatorId` = `employee`.`id`) AS `total`
FROM
  `employee`;

-- 1.4.6 Number 1 and 2
SELECT `name`, `surname`, `email`,
((SELECT COUNT(*) FROM `status` WHERE `creatorId` = `employee` . `id`) +
(SELECT COUNT(*) FROM `comment` WHERE `creatorId` = `employee` . `id`)) AS Total
FROM `employee`
ORDER BY `Total` DESC LIMIT 2;

-- 1.4.7 Most Controversial Employee.
SELECT `employee`.*
FROM (
	SELECT DISTINCT
        (`creatorId`) AS `creatorId`,
        (SELECT COUNT(*)
         FROM `comment`
         WHERE `comment`.`statusId` = `status`.`id`
				) AS `count`
   FROM `status`
   HAVING `count` =
	 		(SELECT COUNT(*)
        FROM
            `comment`
        GROUP BY `statusId`
        ORDER BY `COUNT(*)` DESC
        LIMIT 1
			)
) AS allTable
JOIN `employee` ON `employee`.`id` = `creatorId`
