-- Part 1.1 schema.sql
--
-- Submitted by: Sari Nusier
--

-- -----------------------------------------------------
-- Table `employee`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `employee` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(30) NOT NULL,
  `surname` VARCHAR(30) NOT NULL,
  `sex` ENUM('Male', 'Female') NOT NULL,
  `dateOfBirth` DATE NULL DEFAULT NULL,
  `email` VARCHAR(30) NOT NULL,
  UNIQUE (`email`),
  PRIMARY KEY (`id`)
);

-- -----------------------------------------------------
-- Table `event`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `event` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `creatorId` INT NOT NULL,
  `type` ENUM('Social', 'Business') NOT NULL,
  `description` VARCHAR(200) NOT NULL,
  `date` DATE NOT NULL,
  `time` TIME NOT NULL,
  `location` VARCHAR(100) NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_event_creator`
    FOREIGN KEY (`creatorId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);


-- -----------------------------------------------------

-- Table `status`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `status` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `creatorId` INT NOT NULL,
  `text` VARCHAR(200) NOT NULL,
  `timestamp` TIMESTAMP NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_status_creator`
    FOREIGN KEY (`creatorId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);


-- -----------------------------------------------------
-- Table `comment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `comment` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `creatorId` INT NOT NULL,
  `statusId` INT NOT NULL,
  `text` VARCHAR(200) NOT NULL,
  `timestamp` TIMESTAMP NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_comment_creator`
    FOREIGN KEY (`creatorId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_comment_status`
    FOREIGN KEY (`statusId`)
    REFERENCES `status` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);


-- -----------------------------------------------------
-- Table `likeStatus`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `likeStatus` (
  `employeeId` INT NOT NULL,
  `statusId` INT NOT NULL,
  PRIMARY KEY (`employeeId`, `statusId`),
  CONSTRAINT `fk_likeStatus_status`
    FOREIGN KEY (`statusId`)
    REFERENCES `status` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_likeStatus_employee`
    FOREIGN KEY (`employeeId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);


-- -----------------------------------------------------
-- Table `likeComment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `likeComment` (
  `employeeId` INT NOT NULL,
  `commentId` INT NOT NULL,
  PRIMARY KEY (`employeeId`, `commentId`),
  CONSTRAINT `fk_likeComment_comment`
    FOREIGN KEY (`commentId`)
    REFERENCES `comment` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_likeComment_employee`
    FOREIGN KEY (`employeeId`)
    REFERENCES `employee` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);
