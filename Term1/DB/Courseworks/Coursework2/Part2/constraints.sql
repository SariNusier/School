-- Part 2.1 constraints.sql
--
-- Submitted by: Sari Nusier
--


--  Write your CREATE TRIGGERS statements here
delimiter $$
  create trigger `trigger_check_reported_crime_datetime` before insert on `crimes2015`
  for each row
  begin
    declare error_msg varchar(128);
	  if STR_TO_DATE(NEW.date_occ, '%m/%d/%Y') > STR_TO_DATE(NEW.date_reported, '%m/%d/%Y') then
      set error_msg = 'Crimes cannot be reported before they occur.”';
      signal sqlstate '45000' set message_text = error_msg;
    elseif STR_TO_DATE(NEW.date_occ, '%m/%d/%Y') = CURDATE() then
      if CONVERT(NEW.time_occ, TIME) > CURTIME() then
        set error_msg = 'Crimes cannot be reported before they occur.”';
        signal sqlstate '45000' set message_text = error_msg;
      end if;
    end if;
  end$$
delimiter ;

--  Write your testing statements here

-- Check if date_occ is in the future
INSERT INTO crimes2015 (date_reported, date_occ) VALUES('01/01/2015','01/02/2015');

-- Check if date_occ is today and the time is later than now.
INSERT INTO crimes2015 (date_occ, time_occ) VALUES(date_format(curdate(), '%m/%d/%Y'), curtime()+1);
