import hudson.model.*;
import jenkins.model.*;
import hudson.slaves.*;
import jenkins.*;

def date = new Date()
def no = date.format('yyyyMMddHHmm')
def name = "jenkins_slave_name-$no"
Jenkins.instance.addNode(
  new DumbSlave(
    name,
    "this is a slave node",
    "/opt/work/jenkins",
    "jenkins_slave_executors",
    Node.Mode.NORMAL,
    "jenkins_slave_label",
    new JNLPLauncher(),
    new RetentionStrategy.Always(),
    new LinkedList()
  )
)

def slaves = hudson.model.Hudson.instance.slaves
def slave = slaves.find { it.name == name }
def token = slave.getComputer().getJnlpMac()
println "$token $name"

