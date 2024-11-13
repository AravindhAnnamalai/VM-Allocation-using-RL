import numpy as np

class PhysicalMachine:
    def __init__(self, pm_id, total_cpu, total_memory, energy_per_cpu=0.1):
        self.pm_id = pm_id
        self.total_cpu = total_cpu
        self.total_memory = total_memory
        self.available_cpu = total_cpu
        self.available_memory = total_memory
        self.energy_per_cpu = energy_per_cpu
        self.vms = []  # List of VMs allocated to this PM

    def allocate_vm(self, vm):
        """Attempts to allocate resources for a VM on this PM"""
        if self.can_allocate(vm):
            self.available_cpu -= vm.cpu_required
            self.available_memory -= vm.memory_required
            self.vms.append(vm)
            vm.allocated = True
            vm.allocated_pm = self.pm_id
            return True
        return False

    def can_allocate(self, vm):
        """Checks if this PM can allocate the required resources to a VM"""
        return self.available_cpu >= vm.cpu_required and self.available_memory >= vm.memory_required

    def get_cpu_utilization(self):
        """Returns CPU utilization as a ratio of used to total CPU"""
        return (self.total_cpu - self.available_cpu) / self.total_cpu

    def get_memory_utilization(self):
        """Returns memory utilization as a ratio of used to total memory"""
        return (self.total_memory - self.available_memory) / self.total_memory

    def get_energy_consumption(self):
        """Returns energy consumption based on used CPU"""
        return (self.total_cpu - self.available_cpu) * self.energy_per_cpu

class VirtualMachine:
    def __init__(self, vm_id, cpu_required, memory_required):
        self.vm_id = vm_id
        self.cpu_required = cpu_required
        self.memory_required = memory_required
        self.allocated = False
        self.allocated_pm = None

class Datacenter:
    def __init__(self):
        self.pms = []
        self.vms = []

    def add_pm(self, pm):
        """Adds a physical machine to the datacenter"""
        self.pms.append(pm)

    def add_vm(self, vm):
        """Adds a virtual machine to the datacenter"""
        self.vms.append(vm)

    def get_total_utilization(self):
        """Returns total CPU and memory utilization across all PMs"""
        total_cpu = sum(pm.total_cpu for pm in self.pms)
        total_memory = sum(pm.total_memory for pm in self.pms)
        used_cpu = sum(pm.total_cpu - pm.available_cpu for pm in self.pms)
        used_memory = sum(pm.total_memory - pm.available_memory for pm in self.pms)
        return used_cpu / total_cpu, used_memory / total_memory

    def get_total_energy_consumption(self):
        """Returns total energy consumption across all PMs"""
        return sum(pm.get_energy_consumption() for pm in self.pms)

    def reset(self):
        """Resets the datacenter by deallocating all VMs from PMs"""
        for pm in self.pms:
            pm.available_cpu = pm.total_cpu
            pm.available_memory = pm.total_memory
            pm.vms = []
        for vm in self.vms:
            vm.allocated = False
            vm.allocated_pm = None
