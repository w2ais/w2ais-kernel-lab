#include "arithmetic.cuh"
#include "memory.cuh"
#include "copy.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Case {
  std::string family,op;
  int warps,chains,size,depth,iterations,samples,variant;
};
Timing* dt;
double* dout;
uint64_t* uout;
uint32_t *copy_source,*copy_destination;

void check(double got,double expected,double tolerance=1e-5) {
  if(!std::isfinite(got) || std::abs(got-expected)>tolerance*std::max(1.,std::abs(expected))) {
    fprintf(stderr,"Validation failed: got %.17g expected %.17g\n",got,expected); std::exit(2);
  }
}
template<class Launch,class Validate>
void collect(const Case& c,cudaFuncAttributes attr,int dynamic_bytes,uint64_t work,
             const char* unit,Launch launch,Validate validate) {
  for(int sample=-2;sample<c.samples;sample++) {
    CUDA(cudaMemset(dt,0,sizeof(Timing)));
    launch(); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    if(sample<0) continue;
    Timing t{}; CUDA(cudaMemcpy(&t,dt,sizeof(t),cudaMemcpyDeviceToHost));
    double error=validate();
    printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%llu,%u,%llu,%s,%d,%zu,%zu,%.9g,%llu\n",
      c.family.c_str(),c.op.c_str(),c.warps,c.chains,c.size,c.depth,c.iterations,c.variant,
      sample,t.cycles,t.sm,(unsigned long long)work,unit,attr.numRegs,
      attr.sharedSizeBytes+dynamic_bytes,attr.localSizeBytes,error,t.nanoseconds);
    fflush(stdout);
  }
}

double scalar_reference(Alu op,int lane,int chain,int steps,int variant=0) {
  double x=.25+.125*((lane+chain)%4);
  if(op==Alu::RCP) {
    x=1.25+((lane+chain)%4);
    for(int i=0;i<std::min(steps,128);i++) x=float(1.f/(float(x)+1.f));
    return x;
  }
  if(op==Alu::EXP2) {
    // exp2(-x) is a contraction around its fixed point. Past 128 steps its
    // FP32 result is stable to much better than the 1e-5 validation tolerance.
    for(int i=0;i<std::min(steps,128);i++) x=float(std::exp2(-float(x)));
    return x;
  }
  if(variant==0) x=steps%2 ? 1.-x : x;
  else {
    for(int i=0;i<std::min(steps,128);i++) {
      x=.5*x+1.;
      if(op==Alu::HFMA2) x=__half2float(__float2half_rn(float(x)));
      else if(op==Alu::FFMA) x=float(x);
    }
  }
  return op==Alu::HFMA2 ? 2*x : x;
}

template<Alu op,int N>
void run_arithmetic(const Case& c) {
  cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,arithmetic<op,N>));
  auto launch=[&]{ arithmetic<op,N><<<1,c.warps*32>>>(c.iterations,c.variant ? .5 : -1.,dt,dout); };
  auto validate=[&]{
    std::vector<double> result(c.warps*32);
    CUDA(cudaMemcpy(result.data(),dout,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
    double error=0;
    for(int lane=0;lane<c.warps*32;lane++) {
      double expected=0;
      for(int j=0;j<N;j++) expected+=scalar_reference(op,lane,j,c.iterations*16,c.variant);
      check(result[lane],expected,op==Alu::DFMA ? 1e-12 : 1e-5);
      error=std::max(error,std::abs(result[lane]-expected));
    }
    return error;
  };
  collect(c,a,0,uint64_t(c.warps)*N*16*c.iterations,
          op==Alu::RCP ? "warp_rcp_fadd_pair" : "warp_instruction",launch,validate);
}
template<Alu op> void alu_chains(const Case& c) {
  switch(c.chains) {
    case 1:return run_arithmetic<op,1>(c); case 2:return run_arithmetic<op,2>(c);
    case 4:return run_arithmetic<op,4>(c); case 8:return run_arithmetic<op,8>(c);
    default:std::exit(1);
  }
}
template<Alu op,int R,int Mode> void run_mixed(const Case& c) {
  cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,mixed<op,R,Mode>));
  auto launch=[&]{ mixed<op,R,Mode><<<1,512>>>(c.iterations,-1.f,dt,dout); };
  auto validate=[&]{
    std::vector<double> result(1024);
    CUDA(cudaMemcpy(result.data(),dout,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
    double error=0;
    for(int lane=0;lane<512;lane++) {
      double m=0,s=0;
      for(int j=0;j<4;j++) {
        if(Mode!=1) for(int k=0;k<4;k++) m+=j+k+c.iterations*16*16;
        if(Mode!=0) for(int r=0;r<R;r++) s+=scalar_reference(op,lane,j+r,c.iterations*16);
      }
      check(result[2*lane],m,1e-12); check(result[2*lane+1],s);
      error=std::max(error,std::abs(result[2*lane+1]-s));
    }
    return error;
  };
  collect(c,a,0,uint64_t(16)*4*16*c.iterations,"base_iteration",launch,validate);
}
template<Alu op,int R> void mixed_mode(const Case& c) {
  if(c.variant==0) return run_mixed<op,R,0>(c);
  if(c.variant==1) return run_mixed<op,R,1>(c);
  if(c.variant==2) return run_mixed<op,R,2>(c);
  std::exit(1);
}
template<Alu op> void mixed_ratio(const Case& c) {
  if(c.depth==1) return mixed_mode<op,1>(c);
  if(c.depth==4) return mixed_mode<op,4>(c);
  if(c.depth==8) return mixed_mode<op,8>(c);
  std::exit(1);
}

struct Chain { uint64_t* device; std::vector<uint32_t> order; };
std::map<int,Chain> chains;
Chain& memory_chain(int bytes) {
  if(chains.count(bytes)) return chains.at(bytes);
  Chain chain{}; CUDA(cudaMalloc(&chain.device,bytes));
  int nodes=bytes/128;
  chain.order.resize(nodes); std::iota(chain.order.begin(),chain.order.end(),0);
  std::mt19937 random(20261105u+bytes);
  std::shuffle(chain.order.begin(),chain.order.end(),random);
  std::vector<uint64_t> addresses(bytes/8,0);
  for(int i=0;i<nodes;i++) addresses[chain.order[i]*16]=
    reinterpret_cast<uint64_t>(chain.device)+uint64_t(chain.order[(i+1)%nodes])*128;
  CUDA(cudaMemcpy(chain.device,addresses.data(),bytes,cudaMemcpyHostToDevice));
  return chains.emplace(bytes,std::move(chain)).first->second;
}
template<bool CG> void run_memory(const Case& c) {
  auto& chain=memory_chain(c.size);
  uint64_t first=reinterpret_cast<uint64_t>(chain.device)+uint64_t(chain.order[0])*128;
  uint64_t steps=uint64_t(c.iterations)*32;
  uint64_t expected=reinterpret_cast<uint64_t>(chain.device)+uint64_t(chain.order[steps%chain.order.size()])*128;
  cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,pointer_chase<CG>));
  auto launch=[&]{ pointer_chase<CG><<<1,32>>>(first,c.iterations,dt,uout); };
  auto validate=[&]{ uint64_t result; CUDA(cudaMemcpy(&result,uout,8,cudaMemcpyDeviceToHost));
    if(result!=expected) { fprintf(stderr,"Pointer-chain validation failed\n"); std::exit(2); } return 0.; };
  collect(c,a,0,steps,"dependent_load",launch,validate);
}
void run_shared(const Case& c) {
  cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,shared_chase));
  auto launch=[&]{ shared_chase<<<1,32>>>(c.depth,c.iterations,dt,uout); };
  auto validate=[&]{ uint64_t result[32]; CUDA(cudaMemcpy(result,uout,sizeof(result),cudaMemcpyDeviceToHost));
    for(int lane=0;lane<32;lane++) if(result[lane]!=uint64_t(4*lane*c.depth)) {
      fprintf(stderr,"Shared-chain validation failed\n"); std::exit(2); } return 0.; };
  collect(c,a,0,uint64_t(c.iterations)*32,"warp_load",launch,validate);
}
template<int Mode> void run_sync(const Case& c) {
  cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,synchronization<Mode>));
  auto launch=[&]{ synchronization<Mode><<<1,c.warps*32>>>(c.iterations,dt,uout); };
  auto validate=[&]{ uint64_t result; CUDA(cudaMemcpy(&result,uout,8,cudaMemcpyDeviceToHost));
    if(result!=0) std::exit(2); return 0.; };
  collect(c,a,0,uint64_t(c.iterations)*16,"sync_round",launch,validate);
}
template<int Mode> void run_copy(const Case& c) {
  const int bytes=c.size*c.depth;
  std::vector<uint32_t> values(bytes/4);
  for(int i=0;i<bytes/4;i++) values[i]=0x12340000u+i;
  CUDA(cudaMemcpy(copy_source,values.data(),bytes,cudaMemcpyHostToDevice));
  CUDA(cudaMemset(copy_destination,0,bytes));
  CUtensorMap map{};
  uint64_t dims[2]={64,uint64_t(c.size/256*c.depth)},strides[1]={256};
  uint32_t box[2]={64,uint32_t(c.size/256)},element_stride[2]={1,1};
  DRIVER(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT32,2,copy_source,dims,
    strides,box,element_stride,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  int shared=bytes+128;
  CUDA(cudaFuncSetAttribute(copies<Mode>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,copies<Mode>));
  auto launch=[&]{ copies<Mode><<<1,128,shared>>>(map,copy_source,copy_destination,c.size,c.depth,c.iterations,dt); };
  auto validate=[&]{ std::vector<uint32_t> actual(bytes/4);
    CUDA(cudaMemcpy(actual.data(),copy_destination,bytes,cudaMemcpyDeviceToHost));
    if(actual!=values) { fprintf(stderr,"Copy validation failed: %s %d %d\n",c.op.c_str(),c.size,c.depth); std::exit(2); } return 0.; };
  collect(c,a,shared,uint64_t(bytes)*c.iterations,"byte",launch,validate);
}

int main(int argc,char** argv) {
  if(argc!=2) { fprintf(stderr,"Usage: extended PLAN.csv\n"); return 1; }
  if(std::string(argv[1])=="--info") {
    cudaDeviceProp p{}; CUDA(cudaGetDeviceProperties(&p,0));
    printf("{\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
      "\"l2_bytes\":%d,\"shared_bytes_per_sm\":%zu,\"threads_per_sm\":%d}\n",
      p.name,p.major,p.minor,p.multiProcessorCount,p.l2CacheSize,p.sharedMemPerMultiprocessor,p.maxThreadsPerMultiProcessor);
    return 0;
  }
  CUDA(cudaMalloc(&dt,sizeof(Timing))); CUDA(cudaMalloc(&dout,2048*sizeof(double)));
  CUDA(cudaMalloc(&uout,1024*sizeof(uint64_t)));
  CUDA(cudaMalloc(&copy_source,65536)); CUDA(cudaMalloc(&copy_destination,65536));
  std::ifstream plan(argv[1]); if(!plan) return 1;
  printf("family,op,warps,chains,size,depth,iterations,variant,sample,cycles,sm_id,work,unit,registers_per_thread,shared_bytes,local_bytes,max_abs_error,nanoseconds\n");
  std::string line;
  while(std::getline(plan,line)) {
    if(line.empty() || line.rfind("family,",0)==0) continue;
    std::replace(line.begin(),line.end(),',',' '); std::istringstream fields(line); Case c;
    if(!(fields>>c.family>>c.op>>c.warps>>c.chains>>c.size>>c.depth>>c.iterations>>c.samples>>c.variant)) return 1;
    if(c.warps<1 || c.warps>32 || c.iterations<1 || c.samples<1) return 1;
    if(c.family=="alu") {
      if(c.op=="ffma") alu_chains<Alu::FFMA>(c);
      else if(c.op=="dfma") alu_chains<Alu::DFMA>(c);
      else if(c.op=="hfma2") alu_chains<Alu::HFMA2>(c);
      else if(c.op=="rcp") alu_chains<Alu::RCP>(c);
      else if(c.op=="exp2") alu_chains<Alu::EXP2>(c); else return 1;
    } else if(c.family=="mixed") {
      if(c.warps!=16 || c.chains!=4) return 1;
      if(c.op=="ffma") mixed_ratio<Alu::FFMA>(c);
      else if(c.op=="exp2") mixed_ratio<Alu::EXP2>(c); else return 1;
    } else if(c.family=="memory") {
      if(c.size<4096 || c.size>268435456 || c.size%128) return 1;
      if(c.op=="ca") run_memory<false>(c); else if(c.op=="cg") run_memory<true>(c); else return 1;
    } else if(c.family=="shared") {
      if(c.depth<0 || c.depth>32) return 1;
      run_shared(c);
    } else if(c.family=="sync") {
      if(c.op=="cta") run_sync<0>(c); else if(c.op=="mbarrier") run_sync<1>(c);
      else if(c.op=="fence") run_sync<2>(c); else return 1;
    } else if(c.family=="copy") {
      if(c.depth<1 || c.depth>8 || c.size<1024 || c.size%256 || c.size*c.depth>65536) return 1;
      if(c.op=="vector") run_copy<0>(c); else if(c.op=="cp_async") run_copy<1>(c);
      else if(c.op=="bulk") run_copy<2>(c); else if(c.op=="tma2d") run_copy<3>(c);
      else if(c.op=="store") run_copy<4>(c); else return 1;
    } else return 1;
  }
  for(auto& item:chains) CUDA(cudaFree(item.second.device));
  CUDA(cudaFree(dt)); CUDA(cudaFree(dout)); CUDA(cudaFree(uout));
  CUDA(cudaFree(copy_source)); CUDA(cudaFree(copy_destination));
}
