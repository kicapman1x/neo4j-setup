##neo4j##
export NEO4J_VERSION='10.1'
export NEO4J_HOME="/home/daddy/apps/neo4j/neo4j-$NEO4J_VERSION"

alias startgraphdb="nohup $NEO4J_HOME/bin/neo4j console > /dev/null &"