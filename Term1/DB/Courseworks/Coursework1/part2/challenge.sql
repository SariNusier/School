-- Part 2.2 challenge.sql
--
-- Submitted by: Sari Nusier
--
--


-- add your SQL statements here

-- In order to attach other types of files to a status, a LONGBLOB field must be added
ALTER TABLE `status`
ADD COLUMN `attachment` LONGBLOB NULL;

-- Invitation table is used to store the invitations sent for specific events.
-- A trigger implemented below makes sure that the invitor id is the as the event creatorId
CREATE TABLE `invitation` (
  `invitorId` INT NOT NULL,
  `inviteeId` INT NOT NULL,
  `eventId` INT NOT NULL,
  PRIMARY KEY (`invitorId`, `inviteeId`, `eventId`),
  CONSTRAINT `fk_invitation_invitor`
    FOREIGN KEY (`invitorId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_invitation_invitee`
    FOREIGN KEY (`inviteeId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_invitation_event`
    FOREIGN KEY (`eventId`)
    REFERENCES `event` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);

CREATE TABLE `attendance` (
  `attendeeId` INT NOT NULL,
  `eventId` INT NOT NULL,
  PRIMARY KEY (`attendeeId`, `eventId`),
  CONSTRAINT `fk_attendance_attendee`
    FOREIGN KEY(`attendeeId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_attendance_event`
    FOREIGN KEY(`eventId`)
    REFERENCES `event` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);

-- The trigger makes sure that only event creators can invite others.
-- Otherwise, display an error.
delimiter $$
  CREATE TRIGGER `trigger_before_insert_invitation` BEFORE INSERT ON `invitation`
  for each row
  begin
      declare error_msg varchar(128);
  	  if new.invitorId != (SELECT `creatorId` FROM `event` WHERE `id` = NEW.eventId) THEN
          set error_msg = 'The event was not created by the invitor';
          signal sqlstate '45000' set message_text = error_msg;
      end if;
  end$$
delimiter ;
