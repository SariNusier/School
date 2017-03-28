-- Part 1.5 update.sql
--
-- Submitted by: Sari Nusier
--


-- add your UPDATE statements here

-- 1.5.1 Update Events
SET SQL_SAFE_UPDATES=0;
UPDATE `event` SET `type` = 'Social' WHERE time > '17:00:00';
SET SQL_SAFE_UPDATES=1;

-- 1.5.2 Update Comments
SET SQL_SAFE_UPDATES=0;
UPDATE `comment` SET `timestamp` = (SELECT `status` . `timestamp` + 1 second FROM status WHERE `id` = `comment` . `statusId`)
WHERE `comment`.`timestamp` < (SELECT `status`.`timestamp` FROM status WHERE `id` = `comment` . `statusId`);
SET SQL_SAFE_UPDATES=1;

-- THE BELOW COMMENTED TRIGGERS ARE USED TO ENSURE THAT THE ABOVE RULES WILL BE ENFORCED
-- FOR DATA INSERTED IN THE FUTURE
-- PLEASE UNCOMMENT TO USE!

-- delimiter $$
--   create trigger `trigger_before_insert_event` before insert on `event`
--   for each row
--   begin
--       declare error_msg varchar(128);
--       if NEW.time > '17:00:00' then
--   		  SET NEW.type = 'Social';
--   	  end if;
--   end$$
--
--   create trigger `trigger_before_insert_comment` before insert on `comment`
--   for each row
--   begin
--       declare error_msg varchar(128);
--       if NEW.timestamp < (SELECT `status`.`timestamp` FROM status WHERE `id` = NEW.statusId) then
--   		  SET NEW.timestamp = (SELECT `status`.`timestamp` + 1 FROM status WHERE `id` = NEW.statusId);
--   	  end if;
--   end$$
-- delimiter ;
