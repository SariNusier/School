-- Part 1.1 schema.sql
--
-- Submitted by: Sari Nusier
--


-- add your CREATE TABLE statements here
CREATE TABLE employee(id INT NOT NULL AUTO_INCREMENT, name VARCHAR(30) NOT NULL,
                      surname VARCHAR(30) NOT NULL, sex CHAR(1) NOT NULL,
                      dateOfBirth DATE, email VARCHAR(30) NOT NULL, PRIMARY KEY (id));
CREATE TABLE status(id INT NOT NULL AUTO_INCREMENT, creatorId , text VARCHAR(30) NOT NULL,
                      timestamp TIMESTAMP NOT NULL, sex CHAR(1) NOT NULL,
                      dateOfBirth DATE, email VARCHAR(30) NOT NULL, PRIMARY KEY (id);
CREATE TABLE comment();
CREATE TABLE likeComment();
CREATE TABLE likeStatus();
CREATE TABLE event();
