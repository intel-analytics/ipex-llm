#!groovy

import hudson.model.*;
import jenkins.model.*;
import hudson.slaves.*;
import hudson.security.*;

Thread.start {
    println "--> setting agent port for jnlp"
    def env = System.getenv()
    int agentPort = env['JENKINS_SLAVE_AGENT_PORT'].toInteger()
    Jenkins.instance.setSlaveAgentPort(agentPort)
    println "--> setting agent port for jnlp... done"
    if (env['PROXY_PORT']?.trim()) {
        def proxyHost = env['PROXY_HOST'].toString()
        int proxyPort = env['PROXY_PORT'].toInteger()
        Jenkins.instance.proxy = new hudson.ProxyConfiguration(proxyHost, proxyPort)
    }
    println "--> set proxy... done"
    def hudsonRealm = new HudsonPrivateSecurityRealm(false)
    hudsonRealm.createAccount('admin','admin')
    Jenkins.instance.setSecurityRealm(hudsonRealm)
    def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
    Jenkins.instance.setAuthorizationStrategy(strategy)
    println "--> create admin... done"
    Jenkins.instance.save()
}

