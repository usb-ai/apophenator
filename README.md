# apophenator

In this project we extract dates of previous examinations of a patient report with a trained model and run them through
the database in order to get the reports of the previous examinations.

### Running Neo4j Browser on port 80
For some reason community edition of Neo4j reports port 80 as already used.
To bypass this issue you can set a rule in iptables.config
```bash
sudo apt-get install iptables-persistent
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8000
sudo /sbin/iptables-save > /etc/iptables/rules.v4
```
Note that I am using port 8000 instead of the default port in this example.

The other thing to note is that the browser will try to connect to the database 
from the machine which is running the browser and not the machine that is serving it.
This means the machine and the port of the database need be reachable from the client machine.



### Neo4j config
The default memory values for Neo4j are dynamic. It is helpful to make some changes.
To get recommended values run 
```bash
neo4j-admin memrec
```

Something like this will come out as a result of the command 
```bash
dbms.memory.heap.initial_size=10200m
dbms.memory.heap.max_size=10200m
dbms.memory.pagecache.size=11200m
```
The other intersting recommendation is this 
```bash
dbms.jvm.additional=-XX:+ExitOnOutOfMemoryError
```

This will crash the app when the heap out of memory.

### Fancy additions

If GDS from Neo4j is needed and the server is remote, in order to install it 
follow the [link](https://neo4j.com/docs/graph-data-science/current/installation/neo4j-server/). If it is local 
just do it from the Desktop app.

(See below or link for more details)
```bash
sudo service neo4j stop
wget https://s3-eu-west-1.amazonaws.com/com.neo4j.graphalgorithms.dist/graph-data-science/neo4j-graph-data-science-1.8.6-standalone.zip
sudo apt install unzip
unzip neo4j-graph-data-science-1.8.6-standalone.zip 
sudo cp neo4j-graph-data-science-1.8.6.jar /var/lib/neo4j/plugins
sudo chown --reference=/var/lib/neo4j/plugins/README.txt /var/lib/neo4j/plugins/neo4j-graph-data-science-1.8.6.jar
sudo service neo4j start
```

It can happen that Neo4j enters the crash loop. Could be checked with:
`watch -n 1 sudo service neo4j status`

In which case stop the service and simply `sudo reboot`.

If APOC is needed - same procedure [link](https://neo4j.com/labs/apoc/4.3/installation/#neo4j-server).

Example for APOC
```bash
wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.1/apoc-4.4.0.1-all.jar
sudo cp apoc-4.4.0.1-all.jar /var/lib/neo4j/plugins
sudo chown --reference=/var/lib/neo4j/plugins/README.txt /var/lib/neo4j/plugins/apoc-4.4.0.1-all.jar
```
The second command assumes that the file was download to user's home dir.

I am doing chown because there were some issues when I root copied the file.
I could not bring Neo4j back up until I restarted. Apart from that,
it makes more sense that .jar belongs to the same group as the rest of the Neo4j files.
I've been experiencing these reset loop issues when I made config changes and added APOC jar,
without stopping the service first.

In config we also need to make sure apoc is allowed
```bash
dbms.security.procedures.unrestricted=gds.*,apoc.*
dbms.security.procedures.allowlist=gds.*,apoc.*
```

We are using `apoc.*` in the second command because the original 
command used in the documentation block the procedure `apoc.meta.schema procedure` that is being used by 
Graph Data Science Playground.

```bash
dbms.security.procedures.allowlist=gds.*,apoc.coll.*,apoc.load.*
```

System reboot seems to be very helpful after these changes.

I also used the newes GDS and it broke everything that GDS Playground is using.
So I went down with the version.
