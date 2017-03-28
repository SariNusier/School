-- Part 1.6 delete.sql
--
-- Submitted by: Sari Nusier
--
--


-- add your DELETE statements here

-- FIRST WE FIND THE MOST CONTROVERSIAL EMPLOYEE
SET @controversialEmail = (
  SELECT `employee`.`email`
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
          FROM `comment`
          GROUP BY `statusId`
          ORDER BY `COUNT(*)` DESC
          LIMIT 1
        )
  ) AS allTable
  JOIN `employee` ON `employee`.`id` = `creatorId` LIMIT 1
);

SET SQL_SAFE_UPDATES=0;
DELETE FROM `employee` WHERE `email` = @controversialEmail;
SET SQL_SAFE_UPDATES=1;
