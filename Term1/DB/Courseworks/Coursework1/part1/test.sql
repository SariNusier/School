
-- -----------------------------------------------------
-- Table `employee`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `employee` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(30) NOT NULL,
  `surname` VARCHAR(30) NOT NULL,
  `sex` CHAR(1) NOT NULL,
  `dateOfBirth` DATE NULL DEFAULT NULL,
  `email` VARCHAR(30) NOT NULL,
  PRIMARY KEY (`id`));


-- -----------------------------------------------------
-- Table `event`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `event` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `creatorId` INT NOT NULL,
  `type` ENUM('Social', 'Business') NOT NULL,
  `description` VARCHAR(100) NOT NULL,
  `date` DATE NOT NULL,
  `time` TIME NOT NULL,
  `location` VARCHAR(45) NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_event_1`
    FOREIGN KEY (`creatorId`)
    REFERENCES `employee` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);


-- -----------------------------------------------------
-- Table `status`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `status` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `creatorID` INT NOT NULL,
  `text` VARCHAR(45) NOT NULL,
  `timestamp` TIMESTAMP NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_status_1`
    FOREIGN KEY (`creatorID`)
    REFERENCES `employee` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);


-- -----------------------------------------------------
-- Table `comment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `comment` (
  `id` INT NOT NULL,
  `creatorId` INT NOT NULL,
  `statusId` INT NOT NULL,
  `text` VARCHAR(45) NOT NULL,
  `timestamp` TIMESTAMP NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_comment_1`
    FOREIGN KEY (`creatorId`)
    REFERENCES `employee` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_comment_2`
    FOREIGN KEY (`statusId`)
    REFERENCES `status` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);


-- -----------------------------------------------------
-- Table `likeStatus`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `likeStatus` (
  `employeeId` INT NOT NULL,
  `statusId` INT NOT NULL,
  PRIMARY KEY (`employeeId`, `statusId`),
  CONSTRAINT `fk_likeStatus_1`
    FOREIGN KEY (`statusId`)
    REFERENCES `status` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_likeStatus_2`
    FOREIGN KEY (`employeeId`)
    REFERENCES `employee` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);


-- -----------------------------------------------------
-- Table `likeComment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `likeComment` (
  `employeeId` INT NOT NULL,
  `commentId` INT NOT NULL,
  PRIMARY KEY (`employeeId`, `commentId`),
  CONSTRAINT `fk_likeComment_1`
    FOREIGN KEY (`commentId`)
    REFERENCES `comment` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_likeComment_2`
    FOREIGN KEY (`employeeId`)
    REFERENCES `employee` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
