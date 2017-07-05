## Container ##

Container is a subclass of abstract class AbstractModule, which
declares methods defined in all containers. A container usually
contains some other modules in the `modules` variable. It overrides
many module methods such that calls are propogated to the contained
modules.
