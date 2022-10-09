import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *
from .BaGTI.src.utils import *
from .BaGTI.src.opt import *
import dgl

class CNNScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		dtl = data_type.split('_')
		data_type = '_'.join(dtl[:-2])+'CNN_'+dtl[-2]+'_'+dtl[-1]
		#print("data_type:")
		#print(data_type)
		self.model = eval(data_type+"()")
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		dtl = data_type.split('_')
		print("load_"+'_'.join(dtl[:-2])+"_data("+dtl[-1]+")")
		_, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-2])+"_data("+dtl[-1]+")")

	def run_GOBI(self):
        
        # GET THE HOST CPU VALUES
        host_cpu = [host.getCPU()/100 for host in self.env.hostlist]
        host_cpu_RGB = [int(x*255/100) for x in cpu]

        # GET HOST IDS
        host_ids = [host.id for host in self.env.hostlist]

        # GET THE NUMBER OF CONTAINERS
        host_container_number = []
        for host_id in host_ids:
            host_container_number.append(len(self.env.getContainersOfHost(host_id)))

        host_container_number_RGB = [(int(x*255/10) if x <= 10 else 255) for x in host]
        
        # GET THE POWER
        host_power = [host.getPower() for host in self.env.hostlist]
        host_power_RGB = [int(x*255/100) for x in host_power]


        # GET THE IPS RATIO
        host_ips_ratio = [host.getBaseIPS()/host.ipsCap for host in self.env.hostlist]
        host_ips_ratio_RGB = [int(x*255) for x in host_ips_ratio]

        # GET THE RAM RATIO
        host_ram_ratio = [host.getCurrentRAM()[0]/(host.getCurrentRAM()[0] + host.getRAMAvailable()[0]) for host in self.env.hostlist]
        host_ram_ratio_RGB = [int(x*255) for x in host_ram_ratio]

        # GET THE DISK RATIO
        host_disk_ratio = [host.getCurrentDisk()[0]/host.diskCap.size for host in self.env.hostlist]
        host_disk_ratio_RGB = [int(x*255) for x in host_disk_ratio]

        host_array = np.array([*host_cpu_RGB, *host_container_number_RGB, *host_power_RGB, *host_ips_ratio_RGB, *host_ram_ratio_RGB, *host_disk_ratio_RGB])

        # THE HOST IMAGE HAS BEEN CREATED.
        # NOW THE CONTAINER IMAGE NEEDS TO BE CREATED
        
        # GET WHICH CONTAINERS ARE ALLOCATED TO WHICH HOST
        host_alloc = [(c.getHostID() if c else -1) for c in self.env.containerlist]
        
        # GET RATIO OF THE CONTAINER IPS WRT TO HOST CAPACITY
        host_ipscap = [host.ipsCap for host in self.env.hostlist]

        container_ips = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]
        container_ips_RGB = [int(x*255/host_ipscap[i]) if != -1 else 0 for i, x in zip(host_alloc, container_ips)]

        # GET THE RAM RATIO
        host_ram_size = [host.ramCap.size for host in self.env.containerlist]

        container_ram = [c.getRAM()[0] if c else 0 for c in self.env.containerlist]
        container_ram_RGB = [int(x*255/host_ram_size[i]) if i != -1 else 0 for i, x in zip(host_alloc, container_ram)]

        # GET THE DISK RATIO
        host_disk_size = [host.diskCap.size for host in self.env.hostlist]

        container_disk = [c.getDisk()[0] if c else 0 for c in self.env.containerlist]
        container_disk_RGB = [int(x*255/host_disk_size[i]) if != -1 else 0 for i, x in zip(host_alloc, container_disk)]
        
        # GET THE HOST_ALLOC IMAGE
        host_alloc_RGB = [int((x+1)*255/4) for x in host_alloc]

        # GET CONTAINER CREATION IDS
        container_creation_ids = [c.id if c else -1 for c in self.env.containerlist]

        host_array = np.array([
            *host_cpu_RGB,
            *host_container_number_RGB,
            *host_power_RGB,
            *host_ips_ratio_RGB,
            *host_ram_ratio_RGB,
            *host_disk_ratio_RGB
        ])

        host_array = host_array.reshape(8,3)

        container_array = np.array([
            *container_ips_RGB,
            *container_ram_RGB,
            *container_disk_RGB,
            *host_alloc_RGB
        ])

        container_array = container_array.reshape(8, 10)

        init = np.concatenate((host_array, container_array), axis = 1)
		
        alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: 
				oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)

		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = optCNN(init, self.model, [], self.data_type)
		decision = []

        host_alloc_new = torch.flatten(result[6:7,3:12]).tolist()
        host_alloc_new = [int((x+1)/64) for x in host_alloc_new]

        for i, cid in enumerate(container_creation_ids):
            if cid == -1:
                continue
            if host_alloc[i] == host_alloc_new[i]:
                continue
            elif host_alloc[i] != host_alloc_new[i]:
                decision.append((cid, host_alloc_new[i]))
                
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_GOBI()
		return decision