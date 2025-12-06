##neo4j##
export NEO4J_VERSION='10.1'
export NEO4J_HOME="/home/daddy/apps/neo4j/neo4j-$NEO4J_VERSION"

export neo4j_username=$(cat $SECRETS_DIR/neo4j | grep -i 'neo4j-user' | cut -d':' -f2)
export neo4j_password=$(cat $SECRETS_DIR/neo4j | grep -i 'neo4j-password' | cut -d':' -f2)

alias startgraphdb="nohup $NEO4J_HOME/bin/neo4j console > /dev/null &"
alias startcyphsh="export JAVA_HOME=/home/daddy/apps/neo4j/jdk-21.0.8 && $NEO4J_HOME/bin/cypher-shell -a bolt+s://neo4j.han.gg:7687 -u $neo4j_username -p $neo4j_password"