A Ray VM cluster is required [On-premise cluster documentation](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html)

Currently still in development. Not complete, issues with ray.init().

Steps:
Create VMs (1 head + workers)
Set up network so that worker nodes can reach head node

> Head node should run: ``ray start --head --port=6379 --dashboard-host=0.0.0.0``
> The last argument is important for the dashboard (default port 8265) to be accessible from outside nodes.

> Worker nodes should run: ``ray start --address=<head IP>:6379``

Set up connection to head node and test whether dashboard is visible

Ensure exposition of port ``10001`` on the head node.

``ray[tune]==2.1.0`` and Python``==3.7.7`` are recommended here,
use the same python + ray version on all nodes and the master.