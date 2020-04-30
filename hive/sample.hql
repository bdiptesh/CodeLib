-- ============================================================================
-- Hive query to fetch item sales information
--
-- Environment: hadoop
--
-- Objective: To fetch required info
--
-- Output: ino_spd_lzn.str_attr
--
-- Author: Diptesh.Basak
--
-- Version: 0.1.0
--
-- Date: Apr 08, 2019
--
-- License: BSD 3-Clause
-- ============================================================================

-- Configuration properties for performance optimization

SET mapreduce.job.queuename=etl;
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.parallel=true;
SET hive.exec.reducers.bytes.per.reducer=100000000;
SET hive.auto.convert.join.noconditionaltask.size=25000000;
SET hive.exec.parallel.thread.number=32;
SET mapreduce.job.reduces=-1;
SET hive.exec.max.dynamic.partitions=9999999;
SET hive.exec.max.dynamic.partitions.pernode=999999;
SET mapreduce.map.output.compress=true;
SET hive.merge.mapredfiles=true;
SET hive.merge.tezfiles=true;
SET hive.execution.engine=tez;
SET tez.am.resource.memory.mb=4096;
SET tez.am.java.opts=-server -Xmx3276m -Djava.net.preferIPv4Stack=true -XX:+UseNUMA -XX:+UseParallelGC;
SET hive.tez.container.size=4096;
SET hive.tez.java.opts=-server -Xmx3276m -Djava.net.preferIPv4Stack=true -XX:+UseNUMA -XX:+UseParallelGC;

-- Main

drop table if exists sls_ly;
create temporary table sls_ly as
 select a, b, c
 from ino_spd_fnd.sls_div_dim
 where
 (c BETWEEN '${hivevar:ly_start_d}' AND '${hivevar:ly_end_d}')
 and b in (1, 3)
 group by
 a, b;

DROP TABLE IF EXISTS ino_spd_lzn.str_attr;
create table ino_spd_lzn.str_attr (
 a int COMMENT 'Insert comment here',
 b string COMMENT 'Insert comment here',
 c int COMMENT 'Insert comment here'
 )
COMMENT 'Insert table comment here'
STORED AS TEXTFILE;

INSERT INTO ino_spd_lzn.str_attr
 select
   a,
   b,
   c
from sls_ly;
