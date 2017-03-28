-- Part 1.2 insert.sql
--
-- Submitted by: Sari Nusier
--


-- 3 Celebrities as employees
INSERT INTO `employee` VALUES(1, 'Liam', 'Neeson', 'Male', '1952-06-07', 'liam@neeson.com');
INSERT INTO `employee` VALUES(2, 'Denzel', 'Washington', 'Male', '1954-12-28', 'denzel@washington.com');
INSERT INTO `employee` VALUES(3, 'Justin', 'Bieber', 'Female', '1994-03-1', 'justin@bieber.com');
INSERT INTO `employee` VALUES(4, 'Sari', 'Nusier', 'Male', '1994-12-08', 'sari.nusier@kcl.ac.uk');

-- 3 Statuses per employee
INSERT INTO `status` VALUES (
  1,
  1,
  'Hello everyone, first status EVER!',
  '20160101090012'
);

INSERT INTO `status` VALUES (
  2,
  1,
  'Today was an AMAZING day!',
  '20160101190204'
);

INSERT INTO `status` VALUES (
  3,
  1,
  'Good morning, everyone!',
  '20160102091615'
);

--

INSERT INTO `status` VALUES (
  4,
  2,
  'Thought of trying out this new social network, it looks interesting.',
  '20160102113015'
);

INSERT INTO `status` VALUES (
  5,
  2,
  'Today we are starting to film my new movie. Stay tuned!',
  '20160110071615'
);

INSERT INTO `status` VALUES (
  6,
  2,
  'Filming is going great so far!',
  '20160120092312'
);

--

INSERT INTO `status` VALUES (
  7,
  3,
  'This social network is useless',
  '20160112101336'
);

INSERT INTO `status` VALUES (
  8,
  3,
  'Going to record another song. I am cool!',
  '20160102100313'
);

INSERT INTO `status` VALUES (
  9,
  3,
  'My new song is almost done!',
  '20160103101009'
);



-- 2 Comments per status

INSERT INTO `comment` VALUES (
  1,
  2,
  1,
  'Is this the first comment then?',
  '20160101090503'
);


INSERT INTO `comment` VALUES (
  2,
  3,
  1,
  'Why is this a big deal?',
  '20160101090715'
);

INSERT INTO `comment` VALUES (
  3,
  2,
  2,
  'Wow, glad to hear it!',
  '20160101190703'
);

INSERT INTO `comment` VALUES (
  4,
  3,
  2,
  'Meh, mine was better.',
  '20160101190907'
);

INSERT INTO `comment` VALUES (
  5,
  2,
  3,
  'Morning!',
  '20160102091712'
);

INSERT INTO `comment` VALUES (
  6,
  3,
  3,
  'good morning',
  '20160102091903'
);

INSERT INTO `comment` VALUES (
  7,
  1,
  4,
  'Glad you started posting!',
  '20160102113604'
);

INSERT INTO `comment` VALUES (
  8,
  3,
  4,
  'Another one joining..',
  '20160102114108'
);

INSERT INTO `comment` VALUES (
  9,
  1,
  5,
  'Can\'t wait to hear more about it!',
  '20160110091206'
);

INSERT INTO `comment` VALUES (
  10,
  3,
  5,
  'Sounds like fun.',
  '20160110101348'
);

INSERT INTO `comment` VALUES (
  11,
  1,
  6,
  'Great to hear that!',
  '20160120092712'
);


INSERT INTO `comment` VALUES (
  12,
  3,
  6,
  'congrats!',
  '20160120093112'
);

INSERT INTO `comment` VALUES (
  13,
  1,
  7,
  'What\'s your problem?',
  '20160112101636'
);

INSERT INTO `comment` VALUES (
  14,
  2,
  7,
  'This guy is crazy...',
  '20160112101736'
);

INSERT INTO `comment` VALUES (
  15,
  3,
  7,
  'I\'m not. you are all idiots',
  '20160112101836'
);

INSERT INTO `comment` VALUES (
  16,
  1,
  7,
  'I will find you.',
  '20160112101936'
);

INSERT INTO `comment` VALUES (
  17,
  2,
  7,
  'Hahaha',
  '20160112101836'
);

INSERT INTO `comment` VALUES (
  18,
  1,
  7,
  'You can join me Denzel :D',
  '20160112102136'
);

INSERT INTO `comment` VALUES (
  19,
  2,
  7,
  'Sure',
  '20160112102236'
);


INSERT INTO `comment` VALUES (
  20,
  1,
  7,
  'This status is so messed up it has a comment in the PAST',
  '20140101000000'
);

INSERT INTO `comment` VALUES (
  21,
  1,
  8,
  'Yes, soo cool...',
  '20160102100512'
);

INSERT INTO `comment` VALUES (
  22,
  2,
  8,
  'Good luck!',
  '20160102100753'
);

INSERT INTO `comment` VALUES (
  23,
  1,
  9,
  'Good job!',
  '20160103101203'
);

INSERT INTO `comment` VALUES (
  24,
  2,
  9,
  'I hope it goes well :).',
  '20160103101307'
);


-- 4 Likes per employee
INSERT INTO `likeComment` VALUES (
  1,
  1
);

INSERT INTO `likeComment` VALUES (
  1,
  2
);

INSERT INTO `likeStatus` VALUES (
  1,
  4
);

INSERT INTO `likeStatus` VALUES (
  1,
  5
);

--

INSERT INTO `likeComment` VALUES (
  2,
  4
);

INSERT INTO `likeComment` VALUES (
  2,
  5
);

INSERT INTO `likeStatus` VALUES (
  2,
  1
);

INSERT INTO `likeStatus` VALUES (
  2,
  2
);

--

INSERT INTO `likeComment` VALUES (
  3,
  9
);

INSERT INTO `likeComment` VALUES (
  3,
  10
);

INSERT INTO `likeStatus` VALUES (
  3,
  3
);

INSERT INTO `likeStatus` VALUES (
  3,
  6
);


-- 3 business events January and February 2016
INSERT INTO `event` VALUES (
  1,
  1,
  'Social',
  'Cool event',
  '2016-01-12',
  '13:11:00',
  'Some nice place'
);

INSERT INTO `event` VALUES (
  2,
  2,
  'Social',
  'Cooler event',
  '2016-02-13',
  '18:11:00',
  'Some nice place'
);

INSERT INTO `event` VALUES (
  3,
  3,
  'Social',
  'Very cool event',
  '2016-01-25',
  '11:11:00',
  'Some nice place'
);
-- 3 Social events January and February 2016

INSERT INTO `event` VALUES (
  4,
  1,
  'Business',
  'Business stuff',
  '2016-02-03',
  '19:11:00',
  'Some nice place'
);

INSERT INTO `event` VALUES (
  5,
  2,
  'Business',
  'Meeting',
  '2016-01-16',
  '09:11:00',
  'Some nice place'
);

INSERT INTO `event` VALUES (
  6,
  3,
  'Business',
  'More meetings',
  '2016-02-11',
  '11:11:00',
  'Some nice place'
);

INSERT INTO `event` VALUES (
  7,
  4,
  'Business',
  'Meeting in the future',
  '2020-02-11',
  '11:11:00',
  'London'
);
