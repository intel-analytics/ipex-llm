! Generated automatically.  DO NOT EDIT!

  integer, parameter :: C_FFTW_R2R_KIND = C_INT32_T

  integer(C_INT), parameter :: FFTW_R2HC = 0
  integer(C_INT), parameter :: FFTW_HC2R = 1
  integer(C_INT), parameter :: FFTW_DHT = 2
  integer(C_INT), parameter :: FFTW_REDFT00 = 3
  integer(C_INT), parameter :: FFTW_REDFT01 = 4
  integer(C_INT), parameter :: FFTW_REDFT10 = 5
  integer(C_INT), parameter :: FFTW_REDFT11 = 6
  integer(C_INT), parameter :: FFTW_RODFT00 = 7
  integer(C_INT), parameter :: FFTW_RODFT01 = 8
  integer(C_INT), parameter :: FFTW_RODFT10 = 9
  integer(C_INT), parameter :: FFTW_RODFT11 = 10
  integer(C_INT), parameter :: FFTW_FORWARD = -1
  integer(C_INT), parameter :: FFTW_BACKWARD = +1
  integer(C_INT), parameter :: FFTW_MEASURE = 0
  integer(C_INT), parameter :: FFTW_DESTROY_INPUT = 1
  integer(C_INT), parameter :: FFTW_UNALIGNED = 2
  integer(C_INT), parameter :: FFTW_CONSERVE_MEMORY = 4
  integer(C_INT), parameter :: FFTW_EXHAUSTIVE = 8
  integer(C_INT), parameter :: FFTW_PRESERVE_INPUT = 16
  integer(C_INT), parameter :: FFTW_PATIENT = 32
  integer(C_INT), parameter :: FFTW_ESTIMATE = 64
  integer(C_INT), parameter :: FFTW_WISDOM_ONLY = 2097152
  integer(C_INT), parameter :: FFTW_ESTIMATE_PATIENT = 128
  integer(C_INT), parameter :: FFTW_BELIEVE_PCOST = 256
  integer(C_INT), parameter :: FFTW_NO_DFT_R2HC = 512
  integer(C_INT), parameter :: FFTW_NO_NONTHREADED = 1024
  integer(C_INT), parameter :: FFTW_NO_BUFFERING = 2048
  integer(C_INT), parameter :: FFTW_NO_INDIRECT_OP = 4096
  integer(C_INT), parameter :: FFTW_ALLOW_LARGE_GENERIC = 8192
  integer(C_INT), parameter :: FFTW_NO_RANK_SPLITS = 16384
  integer(C_INT), parameter :: FFTW_NO_VRANK_SPLITS = 32768
  integer(C_INT), parameter :: FFTW_NO_VRECURSE = 65536
  integer(C_INT), parameter :: FFTW_NO_SIMD = 131072
  integer(C_INT), parameter :: FFTW_NO_SLOW = 262144
  integer(C_INT), parameter :: FFTW_NO_FIXED_RADIX_LARGE_N = 524288
  integer(C_INT), parameter :: FFTW_ALLOW_PRUNING = 1048576

  type, bind(C) :: fftw_iodim
     integer(C_INT) n, is, os
  end type fftw_iodim
  type, bind(C) :: fftw_iodim64
     integer(C_INTPTR_T) n, is, os
  end type fftw_iodim64

  interface
    type(C_PTR) function fftw_plan_dft(rank,n,in,out,sign,flags) bind(C, name='fftw_plan_dft')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftw_plan_dft
    
    type(C_PTR) function fftw_plan_dft_1d(n,in,out,sign,flags) bind(C, name='fftw_plan_dft_1d')
      import
      integer(C_INT), value :: n
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftw_plan_dft_1d
    
    type(C_PTR) function fftw_plan_dft_2d(n0,n1,in,out,sign,flags) bind(C, name='fftw_plan_dft_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftw_plan_dft_2d
    
    type(C_PTR) function fftw_plan_dft_3d(n0,n1,n2,in,out,sign,flags) bind(C, name='fftw_plan_dft_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftw_plan_dft_3d
    
    type(C_PTR) function fftw_plan_many_dft(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,sign,flags) &
                         bind(C, name='fftw_plan_many_dft')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftw_plan_many_dft
    
    type(C_PTR) function fftw_plan_guru_dft(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
                         bind(C, name='fftw_plan_guru_dft')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim), dimension(*), intent(in) :: howmany_dims
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftw_plan_guru_dft
    
    type(C_PTR) function fftw_plan_guru_split_dft(rank,dims,howmany_rank,howmany_dims,ri,ii,ro,io,flags) &
                         bind(C, name='fftw_plan_guru_split_dft')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: ri
      real(C_DOUBLE), dimension(*), intent(out) :: ii
      real(C_DOUBLE), dimension(*), intent(out) :: ro
      real(C_DOUBLE), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftw_plan_guru_split_dft
    
    type(C_PTR) function fftw_plan_guru64_dft(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
                         bind(C, name='fftw_plan_guru64_dft')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim64), dimension(*), intent(in) :: howmany_dims
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftw_plan_guru64_dft
    
    type(C_PTR) function fftw_plan_guru64_split_dft(rank,dims,howmany_rank,howmany_dims,ri,ii,ro,io,flags) &
                         bind(C, name='fftw_plan_guru64_split_dft')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: ri
      real(C_DOUBLE), dimension(*), intent(out) :: ii
      real(C_DOUBLE), dimension(*), intent(out) :: ro
      real(C_DOUBLE), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftw_plan_guru64_split_dft
    
    subroutine fftw_execute_dft(p,in,out) bind(C, name='fftw_execute_dft')
      import
      type(C_PTR), value :: p
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(inout) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
    end subroutine fftw_execute_dft
    
    subroutine fftw_execute_split_dft(p,ri,ii,ro,io) bind(C, name='fftw_execute_split_dft')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), dimension(*), intent(inout) :: ri
      real(C_DOUBLE), dimension(*), intent(inout) :: ii
      real(C_DOUBLE), dimension(*), intent(out) :: ro
      real(C_DOUBLE), dimension(*), intent(out) :: io
    end subroutine fftw_execute_split_dft
    
    type(C_PTR) function fftw_plan_many_dft_r2c(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) &
                         bind(C, name='fftw_plan_many_dft_r2c')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      real(C_DOUBLE), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: flags
    end function fftw_plan_many_dft_r2c
    
    type(C_PTR) function fftw_plan_dft_r2c(rank,n,in,out,flags) bind(C, name='fftw_plan_dft_r2c')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      real(C_DOUBLE), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_r2c
    
    type(C_PTR) function fftw_plan_dft_r2c_1d(n,in,out,flags) bind(C, name='fftw_plan_dft_r2c_1d')
      import
      integer(C_INT), value :: n
      real(C_DOUBLE), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_r2c_1d
    
    type(C_PTR) function fftw_plan_dft_r2c_2d(n0,n1,in,out,flags) bind(C, name='fftw_plan_dft_r2c_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      real(C_DOUBLE), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_r2c_2d
    
    type(C_PTR) function fftw_plan_dft_r2c_3d(n0,n1,n2,in,out,flags) bind(C, name='fftw_plan_dft_r2c_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      real(C_DOUBLE), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_r2c_3d
    
    type(C_PTR) function fftw_plan_many_dft_c2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) &
                         bind(C, name='fftw_plan_many_dft_c2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: flags
    end function fftw_plan_many_dft_c2r
    
    type(C_PTR) function fftw_plan_dft_c2r(rank,n,in,out,flags) bind(C, name='fftw_plan_dft_c2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_c2r
    
    type(C_PTR) function fftw_plan_dft_c2r_1d(n,in,out,flags) bind(C, name='fftw_plan_dft_c2r_1d')
      import
      integer(C_INT), value :: n
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_c2r_1d
    
    type(C_PTR) function fftw_plan_dft_c2r_2d(n0,n1,in,out,flags) bind(C, name='fftw_plan_dft_c2r_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_c2r_2d
    
    type(C_PTR) function fftw_plan_dft_c2r_3d(n0,n1,n2,in,out,flags) bind(C, name='fftw_plan_dft_c2r_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_dft_c2r_3d
    
    type(C_PTR) function fftw_plan_guru_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftw_plan_guru_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_guru_dft_r2c
    
    type(C_PTR) function fftw_plan_guru_dft_c2r(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftw_plan_guru_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim), dimension(*), intent(in) :: howmany_dims
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_guru_dft_c2r
    
    type(C_PTR) function fftw_plan_guru_split_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,ro,io,flags) &
                         bind(C, name='fftw_plan_guru_split_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: ro
      real(C_DOUBLE), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftw_plan_guru_split_dft_r2c
    
    type(C_PTR) function fftw_plan_guru_split_dft_c2r(rank,dims,howmany_rank,howmany_dims,ri,ii,out,flags) &
                         bind(C, name='fftw_plan_guru_split_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: ri
      real(C_DOUBLE), dimension(*), intent(out) :: ii
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_guru_split_dft_c2r
    
    type(C_PTR) function fftw_plan_guru64_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftw_plan_guru64_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_guru64_dft_r2c
    
    type(C_PTR) function fftw_plan_guru64_dft_c2r(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftw_plan_guru64_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim64), dimension(*), intent(in) :: howmany_dims
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_guru64_dft_c2r
    
    type(C_PTR) function fftw_plan_guru64_split_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,ro,io,flags) &
                         bind(C, name='fftw_plan_guru64_split_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: ro
      real(C_DOUBLE), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftw_plan_guru64_split_dft_r2c
    
    type(C_PTR) function fftw_plan_guru64_split_dft_c2r(rank,dims,howmany_rank,howmany_dims,ri,ii,out,flags) &
                         bind(C, name='fftw_plan_guru64_split_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: ri
      real(C_DOUBLE), dimension(*), intent(out) :: ii
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftw_plan_guru64_split_dft_c2r
    
    subroutine fftw_execute_dft_r2c(p,in,out) bind(C, name='fftw_execute_dft_r2c')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), dimension(*), intent(inout) :: in
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(out) :: out
    end subroutine fftw_execute_dft_r2c
    
    subroutine fftw_execute_dft_c2r(p,in,out) bind(C, name='fftw_execute_dft_c2r')
      import
      type(C_PTR), value :: p
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(inout) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
    end subroutine fftw_execute_dft_c2r
    
    subroutine fftw_execute_split_dft_r2c(p,in,ro,io) bind(C, name='fftw_execute_split_dft_r2c')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), dimension(*), intent(inout) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: ro
      real(C_DOUBLE), dimension(*), intent(out) :: io
    end subroutine fftw_execute_split_dft_r2c
    
    subroutine fftw_execute_split_dft_c2r(p,ri,ii,out) bind(C, name='fftw_execute_split_dft_c2r')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), dimension(*), intent(inout) :: ri
      real(C_DOUBLE), dimension(*), intent(inout) :: ii
      real(C_DOUBLE), dimension(*), intent(out) :: out
    end subroutine fftw_execute_split_dft_c2r
    
    type(C_PTR) function fftw_plan_many_r2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,kind,flags) &
                         bind(C, name='fftw_plan_many_r2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      real(C_DOUBLE), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftw_plan_many_r2r
    
    type(C_PTR) function fftw_plan_r2r(rank,n,in,out,kind,flags) bind(C, name='fftw_plan_r2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftw_plan_r2r
    
    type(C_PTR) function fftw_plan_r2r_1d(n,in,out,kind,flags) bind(C, name='fftw_plan_r2r_1d')
      import
      integer(C_INT), value :: n
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind
      integer(C_INT), value :: flags
    end function fftw_plan_r2r_1d
    
    type(C_PTR) function fftw_plan_r2r_2d(n0,n1,in,out,kind0,kind1,flags) bind(C, name='fftw_plan_r2r_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind0
      integer(C_FFTW_R2R_KIND), value :: kind1
      integer(C_INT), value :: flags
    end function fftw_plan_r2r_2d
    
    type(C_PTR) function fftw_plan_r2r_3d(n0,n1,n2,in,out,kind0,kind1,kind2,flags) bind(C, name='fftw_plan_r2r_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind0
      integer(C_FFTW_R2R_KIND), value :: kind1
      integer(C_FFTW_R2R_KIND), value :: kind2
      integer(C_INT), value :: flags
    end function fftw_plan_r2r_3d
    
    type(C_PTR) function fftw_plan_guru_r2r(rank,dims,howmany_rank,howmany_dims,in,out,kind,flags) &
                         bind(C, name='fftw_plan_guru_r2r')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftw_plan_guru_r2r
    
    type(C_PTR) function fftw_plan_guru64_r2r(rank,dims,howmany_rank,howmany_dims,in,out,kind,flags) &
                         bind(C, name='fftw_plan_guru64_r2r')
      import
      integer(C_INT), value :: rank
      type(fftw_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftw_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_DOUBLE), dimension(*), intent(out) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftw_plan_guru64_r2r
    
    subroutine fftw_execute_r2r(p,in,out) bind(C, name='fftw_execute_r2r')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), dimension(*), intent(inout) :: in
      real(C_DOUBLE), dimension(*), intent(out) :: out
    end subroutine fftw_execute_r2r
    
    subroutine fftw_destroy_plan(p) bind(C, name='fftw_destroy_plan')
      import
      type(C_PTR), value :: p
    end subroutine fftw_destroy_plan
    
    subroutine fftw_forget_wisdom() bind(C, name='fftw_forget_wisdom')
      import
    end subroutine fftw_forget_wisdom
    
    subroutine fftw_cleanup() bind(C, name='fftw_cleanup')
      import
    end subroutine fftw_cleanup
    
    subroutine fftw_set_timelimit(t) bind(C, name='fftw_set_timelimit')
      import
      real(C_DOUBLE), value :: t
    end subroutine fftw_set_timelimit
    
    subroutine fftw_plan_with_nthreads(nthreads) bind(C, name='fftw_plan_with_nthreads')
      import
      integer(C_INT), value :: nthreads
    end subroutine fftw_plan_with_nthreads
    
    integer(C_INT) function fftw_planner_nthreads() bind(C, name='fftw_planner_nthreads')
      import
    end function fftw_planner_nthreads
    
    integer(C_INT) function fftw_init_threads() bind(C, name='fftw_init_threads')
      import
    end function fftw_init_threads
    
    subroutine fftw_cleanup_threads() bind(C, name='fftw_cleanup_threads')
      import
    end subroutine fftw_cleanup_threads
    
! Unable to generate Fortran interface for fftw_threads_set_callback
    subroutine fftw_make_planner_thread_safe() bind(C, name='fftw_make_planner_thread_safe')
      import
    end subroutine fftw_make_planner_thread_safe
    
    integer(C_INT) function fftw_export_wisdom_to_filename(filename) bind(C, name='fftw_export_wisdom_to_filename')
      import
      character(C_CHAR), dimension(*), intent(in) :: filename
    end function fftw_export_wisdom_to_filename
    
    subroutine fftw_export_wisdom_to_file(output_file) bind(C, name='fftw_export_wisdom_to_file')
      import
      type(C_PTR), value :: output_file
    end subroutine fftw_export_wisdom_to_file
    
    type(C_PTR) function fftw_export_wisdom_to_string() bind(C, name='fftw_export_wisdom_to_string')
      import
    end function fftw_export_wisdom_to_string
    
    subroutine fftw_export_wisdom(write_char,data) bind(C, name='fftw_export_wisdom')
      import
      type(C_FUNPTR), value :: write_char
      type(C_PTR), value :: data
    end subroutine fftw_export_wisdom
    
    integer(C_INT) function fftw_import_system_wisdom() bind(C, name='fftw_import_system_wisdom')
      import
    end function fftw_import_system_wisdom
    
    integer(C_INT) function fftw_import_wisdom_from_filename(filename) bind(C, name='fftw_import_wisdom_from_filename')
      import
      character(C_CHAR), dimension(*), intent(in) :: filename
    end function fftw_import_wisdom_from_filename
    
    integer(C_INT) function fftw_import_wisdom_from_file(input_file) bind(C, name='fftw_import_wisdom_from_file')
      import
      type(C_PTR), value :: input_file
    end function fftw_import_wisdom_from_file
    
    integer(C_INT) function fftw_import_wisdom_from_string(input_string) bind(C, name='fftw_import_wisdom_from_string')
      import
      character(C_CHAR), dimension(*), intent(in) :: input_string
    end function fftw_import_wisdom_from_string
    
    integer(C_INT) function fftw_import_wisdom(read_char,data) bind(C, name='fftw_import_wisdom')
      import
      type(C_FUNPTR), value :: read_char
      type(C_PTR), value :: data
    end function fftw_import_wisdom
    
    subroutine fftw_fprint_plan(p,output_file) bind(C, name='fftw_fprint_plan')
      import
      type(C_PTR), value :: p
      type(C_PTR), value :: output_file
    end subroutine fftw_fprint_plan
    
    subroutine fftw_print_plan(p) bind(C, name='fftw_print_plan')
      import
      type(C_PTR), value :: p
    end subroutine fftw_print_plan
    
    type(C_PTR) function fftw_sprint_plan(p) bind(C, name='fftw_sprint_plan')
      import
      type(C_PTR), value :: p
    end function fftw_sprint_plan
    
    type(C_PTR) function fftw_malloc(n) bind(C, name='fftw_malloc')
      import
      integer(C_SIZE_T), value :: n
    end function fftw_malloc
    
    type(C_PTR) function fftw_alloc_real(n) bind(C, name='fftw_alloc_real')
      import
      integer(C_SIZE_T), value :: n
    end function fftw_alloc_real
    
    type(C_PTR) function fftw_alloc_complex(n) bind(C, name='fftw_alloc_complex')
      import
      integer(C_SIZE_T), value :: n
    end function fftw_alloc_complex
    
    subroutine fftw_free(p) bind(C, name='fftw_free')
      import
      type(C_PTR), value :: p
    end subroutine fftw_free
    
    subroutine fftw_flops(p,add,mul,fmas) bind(C, name='fftw_flops')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), intent(out) :: add
      real(C_DOUBLE), intent(out) :: mul
      real(C_DOUBLE), intent(out) :: fmas
    end subroutine fftw_flops
    
    real(C_DOUBLE) function fftw_estimate_cost(p) bind(C, name='fftw_estimate_cost')
      import
      type(C_PTR), value :: p
    end function fftw_estimate_cost
    
    real(C_DOUBLE) function fftw_cost(p) bind(C, name='fftw_cost')
      import
      type(C_PTR), value :: p
    end function fftw_cost
    
    integer(C_INT) function fftw_alignment_of(p) bind(C, name='fftw_alignment_of')
      import
      real(C_DOUBLE), dimension(*), intent(out) :: p
    end function fftw_alignment_of
    
  end interface

  type, bind(C) :: fftwf_iodim
     integer(C_INT) n, is, os
  end type fftwf_iodim
  type, bind(C) :: fftwf_iodim64
     integer(C_INTPTR_T) n, is, os
  end type fftwf_iodim64

  interface
    type(C_PTR) function fftwf_plan_dft(rank,n,in,out,sign,flags) bind(C, name='fftwf_plan_dft')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwf_plan_dft
    
    type(C_PTR) function fftwf_plan_dft_1d(n,in,out,sign,flags) bind(C, name='fftwf_plan_dft_1d')
      import
      integer(C_INT), value :: n
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_1d
    
    type(C_PTR) function fftwf_plan_dft_2d(n0,n1,in,out,sign,flags) bind(C, name='fftwf_plan_dft_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_2d
    
    type(C_PTR) function fftwf_plan_dft_3d(n0,n1,n2,in,out,sign,flags) bind(C, name='fftwf_plan_dft_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_3d
    
    type(C_PTR) function fftwf_plan_many_dft(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,sign,flags) &
                         bind(C, name='fftwf_plan_many_dft')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwf_plan_many_dft
    
    type(C_PTR) function fftwf_plan_guru_dft(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
                         bind(C, name='fftwf_plan_guru_dft')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim), dimension(*), intent(in) :: howmany_dims
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwf_plan_guru_dft
    
    type(C_PTR) function fftwf_plan_guru_split_dft(rank,dims,howmany_rank,howmany_dims,ri,ii,ro,io,flags) &
                         bind(C, name='fftwf_plan_guru_split_dft')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: ri
      real(C_FLOAT), dimension(*), intent(out) :: ii
      real(C_FLOAT), dimension(*), intent(out) :: ro
      real(C_FLOAT), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwf_plan_guru_split_dft
    
    type(C_PTR) function fftwf_plan_guru64_dft(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
                         bind(C, name='fftwf_plan_guru64_dft')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim64), dimension(*), intent(in) :: howmany_dims
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwf_plan_guru64_dft
    
    type(C_PTR) function fftwf_plan_guru64_split_dft(rank,dims,howmany_rank,howmany_dims,ri,ii,ro,io,flags) &
                         bind(C, name='fftwf_plan_guru64_split_dft')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: ri
      real(C_FLOAT), dimension(*), intent(out) :: ii
      real(C_FLOAT), dimension(*), intent(out) :: ro
      real(C_FLOAT), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwf_plan_guru64_split_dft
    
    subroutine fftwf_execute_dft(p,in,out) bind(C, name='fftwf_execute_dft')
      import
      type(C_PTR), value :: p
      complex(C_FLOAT_COMPLEX), dimension(*), intent(inout) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
    end subroutine fftwf_execute_dft
    
    subroutine fftwf_execute_split_dft(p,ri,ii,ro,io) bind(C, name='fftwf_execute_split_dft')
      import
      type(C_PTR), value :: p
      real(C_FLOAT), dimension(*), intent(inout) :: ri
      real(C_FLOAT), dimension(*), intent(inout) :: ii
      real(C_FLOAT), dimension(*), intent(out) :: ro
      real(C_FLOAT), dimension(*), intent(out) :: io
    end subroutine fftwf_execute_split_dft
    
    type(C_PTR) function fftwf_plan_many_dft_r2c(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) &
                         bind(C, name='fftwf_plan_many_dft_r2c')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      real(C_FLOAT), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: flags
    end function fftwf_plan_many_dft_r2c
    
    type(C_PTR) function fftwf_plan_dft_r2c(rank,n,in,out,flags) bind(C, name='fftwf_plan_dft_r2c')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      real(C_FLOAT), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_r2c
    
    type(C_PTR) function fftwf_plan_dft_r2c_1d(n,in,out,flags) bind(C, name='fftwf_plan_dft_r2c_1d')
      import
      integer(C_INT), value :: n
      real(C_FLOAT), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_r2c_1d
    
    type(C_PTR) function fftwf_plan_dft_r2c_2d(n0,n1,in,out,flags) bind(C, name='fftwf_plan_dft_r2c_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      real(C_FLOAT), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_r2c_2d
    
    type(C_PTR) function fftwf_plan_dft_r2c_3d(n0,n1,n2,in,out,flags) bind(C, name='fftwf_plan_dft_r2c_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      real(C_FLOAT), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_r2c_3d
    
    type(C_PTR) function fftwf_plan_many_dft_c2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) &
                         bind(C, name='fftwf_plan_many_dft_c2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: flags
    end function fftwf_plan_many_dft_c2r
    
    type(C_PTR) function fftwf_plan_dft_c2r(rank,n,in,out,flags) bind(C, name='fftwf_plan_dft_c2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_c2r
    
    type(C_PTR) function fftwf_plan_dft_c2r_1d(n,in,out,flags) bind(C, name='fftwf_plan_dft_c2r_1d')
      import
      integer(C_INT), value :: n
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_c2r_1d
    
    type(C_PTR) function fftwf_plan_dft_c2r_2d(n0,n1,in,out,flags) bind(C, name='fftwf_plan_dft_c2r_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_c2r_2d
    
    type(C_PTR) function fftwf_plan_dft_c2r_3d(n0,n1,n2,in,out,flags) bind(C, name='fftwf_plan_dft_c2r_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_dft_c2r_3d
    
    type(C_PTR) function fftwf_plan_guru_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwf_plan_guru_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_guru_dft_r2c
    
    type(C_PTR) function fftwf_plan_guru_dft_c2r(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwf_plan_guru_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim), dimension(*), intent(in) :: howmany_dims
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_guru_dft_c2r
    
    type(C_PTR) function fftwf_plan_guru_split_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,ro,io,flags) &
                         bind(C, name='fftwf_plan_guru_split_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: ro
      real(C_FLOAT), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwf_plan_guru_split_dft_r2c
    
    type(C_PTR) function fftwf_plan_guru_split_dft_c2r(rank,dims,howmany_rank,howmany_dims,ri,ii,out,flags) &
                         bind(C, name='fftwf_plan_guru_split_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: ri
      real(C_FLOAT), dimension(*), intent(out) :: ii
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_guru_split_dft_c2r
    
    type(C_PTR) function fftwf_plan_guru64_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwf_plan_guru64_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_guru64_dft_r2c
    
    type(C_PTR) function fftwf_plan_guru64_dft_c2r(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwf_plan_guru64_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim64), dimension(*), intent(in) :: howmany_dims
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_guru64_dft_c2r
    
    type(C_PTR) function fftwf_plan_guru64_split_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,ro,io,flags) &
                         bind(C, name='fftwf_plan_guru64_split_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: ro
      real(C_FLOAT), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwf_plan_guru64_split_dft_r2c
    
    type(C_PTR) function fftwf_plan_guru64_split_dft_c2r(rank,dims,howmany_rank,howmany_dims,ri,ii,out,flags) &
                         bind(C, name='fftwf_plan_guru64_split_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: ri
      real(C_FLOAT), dimension(*), intent(out) :: ii
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwf_plan_guru64_split_dft_c2r
    
    subroutine fftwf_execute_dft_r2c(p,in,out) bind(C, name='fftwf_execute_dft_r2c')
      import
      type(C_PTR), value :: p
      real(C_FLOAT), dimension(*), intent(inout) :: in
      complex(C_FLOAT_COMPLEX), dimension(*), intent(out) :: out
    end subroutine fftwf_execute_dft_r2c
    
    subroutine fftwf_execute_dft_c2r(p,in,out) bind(C, name='fftwf_execute_dft_c2r')
      import
      type(C_PTR), value :: p
      complex(C_FLOAT_COMPLEX), dimension(*), intent(inout) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
    end subroutine fftwf_execute_dft_c2r
    
    subroutine fftwf_execute_split_dft_r2c(p,in,ro,io) bind(C, name='fftwf_execute_split_dft_r2c')
      import
      type(C_PTR), value :: p
      real(C_FLOAT), dimension(*), intent(inout) :: in
      real(C_FLOAT), dimension(*), intent(out) :: ro
      real(C_FLOAT), dimension(*), intent(out) :: io
    end subroutine fftwf_execute_split_dft_r2c
    
    subroutine fftwf_execute_split_dft_c2r(p,ri,ii,out) bind(C, name='fftwf_execute_split_dft_c2r')
      import
      type(C_PTR), value :: p
      real(C_FLOAT), dimension(*), intent(inout) :: ri
      real(C_FLOAT), dimension(*), intent(inout) :: ii
      real(C_FLOAT), dimension(*), intent(out) :: out
    end subroutine fftwf_execute_split_dft_c2r
    
    type(C_PTR) function fftwf_plan_many_r2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,kind,flags) &
                         bind(C, name='fftwf_plan_many_r2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      real(C_FLOAT), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwf_plan_many_r2r
    
    type(C_PTR) function fftwf_plan_r2r(rank,n,in,out,kind,flags) bind(C, name='fftwf_plan_r2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwf_plan_r2r
    
    type(C_PTR) function fftwf_plan_r2r_1d(n,in,out,kind,flags) bind(C, name='fftwf_plan_r2r_1d')
      import
      integer(C_INT), value :: n
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind
      integer(C_INT), value :: flags
    end function fftwf_plan_r2r_1d
    
    type(C_PTR) function fftwf_plan_r2r_2d(n0,n1,in,out,kind0,kind1,flags) bind(C, name='fftwf_plan_r2r_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind0
      integer(C_FFTW_R2R_KIND), value :: kind1
      integer(C_INT), value :: flags
    end function fftwf_plan_r2r_2d
    
    type(C_PTR) function fftwf_plan_r2r_3d(n0,n1,n2,in,out,kind0,kind1,kind2,flags) bind(C, name='fftwf_plan_r2r_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind0
      integer(C_FFTW_R2R_KIND), value :: kind1
      integer(C_FFTW_R2R_KIND), value :: kind2
      integer(C_INT), value :: flags
    end function fftwf_plan_r2r_3d
    
    type(C_PTR) function fftwf_plan_guru_r2r(rank,dims,howmany_rank,howmany_dims,in,out,kind,flags) &
                         bind(C, name='fftwf_plan_guru_r2r')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwf_plan_guru_r2r
    
    type(C_PTR) function fftwf_plan_guru64_r2r(rank,dims,howmany_rank,howmany_dims,in,out,kind,flags) &
                         bind(C, name='fftwf_plan_guru64_r2r')
      import
      integer(C_INT), value :: rank
      type(fftwf_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwf_iodim64), dimension(*), intent(in) :: howmany_dims
      real(C_FLOAT), dimension(*), intent(out) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwf_plan_guru64_r2r
    
    subroutine fftwf_execute_r2r(p,in,out) bind(C, name='fftwf_execute_r2r')
      import
      type(C_PTR), value :: p
      real(C_FLOAT), dimension(*), intent(inout) :: in
      real(C_FLOAT), dimension(*), intent(out) :: out
    end subroutine fftwf_execute_r2r
    
    subroutine fftwf_destroy_plan(p) bind(C, name='fftwf_destroy_plan')
      import
      type(C_PTR), value :: p
    end subroutine fftwf_destroy_plan
    
    subroutine fftwf_forget_wisdom() bind(C, name='fftwf_forget_wisdom')
      import
    end subroutine fftwf_forget_wisdom
    
    subroutine fftwf_cleanup() bind(C, name='fftwf_cleanup')
      import
    end subroutine fftwf_cleanup
    
    subroutine fftwf_set_timelimit(t) bind(C, name='fftwf_set_timelimit')
      import
      real(C_DOUBLE), value :: t
    end subroutine fftwf_set_timelimit
    
    subroutine fftwf_plan_with_nthreads(nthreads) bind(C, name='fftwf_plan_with_nthreads')
      import
      integer(C_INT), value :: nthreads
    end subroutine fftwf_plan_with_nthreads
    
    integer(C_INT) function fftwf_planner_nthreads() bind(C, name='fftwf_planner_nthreads')
      import
    end function fftwf_planner_nthreads
    
    integer(C_INT) function fftwf_init_threads() bind(C, name='fftwf_init_threads')
      import
    end function fftwf_init_threads
    
    subroutine fftwf_cleanup_threads() bind(C, name='fftwf_cleanup_threads')
      import
    end subroutine fftwf_cleanup_threads
    
! Unable to generate Fortran interface for fftwf_threads_set_callback
    subroutine fftwf_make_planner_thread_safe() bind(C, name='fftwf_make_planner_thread_safe')
      import
    end subroutine fftwf_make_planner_thread_safe
    
    integer(C_INT) function fftwf_export_wisdom_to_filename(filename) bind(C, name='fftwf_export_wisdom_to_filename')
      import
      character(C_CHAR), dimension(*), intent(in) :: filename
    end function fftwf_export_wisdom_to_filename
    
    subroutine fftwf_export_wisdom_to_file(output_file) bind(C, name='fftwf_export_wisdom_to_file')
      import
      type(C_PTR), value :: output_file
    end subroutine fftwf_export_wisdom_to_file
    
    type(C_PTR) function fftwf_export_wisdom_to_string() bind(C, name='fftwf_export_wisdom_to_string')
      import
    end function fftwf_export_wisdom_to_string
    
    subroutine fftwf_export_wisdom(write_char,data) bind(C, name='fftwf_export_wisdom')
      import
      type(C_FUNPTR), value :: write_char
      type(C_PTR), value :: data
    end subroutine fftwf_export_wisdom
    
    integer(C_INT) function fftwf_import_system_wisdom() bind(C, name='fftwf_import_system_wisdom')
      import
    end function fftwf_import_system_wisdom
    
    integer(C_INT) function fftwf_import_wisdom_from_filename(filename) bind(C, name='fftwf_import_wisdom_from_filename')
      import
      character(C_CHAR), dimension(*), intent(in) :: filename
    end function fftwf_import_wisdom_from_filename
    
    integer(C_INT) function fftwf_import_wisdom_from_file(input_file) bind(C, name='fftwf_import_wisdom_from_file')
      import
      type(C_PTR), value :: input_file
    end function fftwf_import_wisdom_from_file
    
    integer(C_INT) function fftwf_import_wisdom_from_string(input_string) bind(C, name='fftwf_import_wisdom_from_string')
      import
      character(C_CHAR), dimension(*), intent(in) :: input_string
    end function fftwf_import_wisdom_from_string
    
    integer(C_INT) function fftwf_import_wisdom(read_char,data) bind(C, name='fftwf_import_wisdom')
      import
      type(C_FUNPTR), value :: read_char
      type(C_PTR), value :: data
    end function fftwf_import_wisdom
    
    subroutine fftwf_fprint_plan(p,output_file) bind(C, name='fftwf_fprint_plan')
      import
      type(C_PTR), value :: p
      type(C_PTR), value :: output_file
    end subroutine fftwf_fprint_plan
    
    subroutine fftwf_print_plan(p) bind(C, name='fftwf_print_plan')
      import
      type(C_PTR), value :: p
    end subroutine fftwf_print_plan
    
    type(C_PTR) function fftwf_sprint_plan(p) bind(C, name='fftwf_sprint_plan')
      import
      type(C_PTR), value :: p
    end function fftwf_sprint_plan
    
    type(C_PTR) function fftwf_malloc(n) bind(C, name='fftwf_malloc')
      import
      integer(C_SIZE_T), value :: n
    end function fftwf_malloc
    
    type(C_PTR) function fftwf_alloc_real(n) bind(C, name='fftwf_alloc_real')
      import
      integer(C_SIZE_T), value :: n
    end function fftwf_alloc_real
    
    type(C_PTR) function fftwf_alloc_complex(n) bind(C, name='fftwf_alloc_complex')
      import
      integer(C_SIZE_T), value :: n
    end function fftwf_alloc_complex
    
    subroutine fftwf_free(p) bind(C, name='fftwf_free')
      import
      type(C_PTR), value :: p
    end subroutine fftwf_free
    
    subroutine fftwf_flops(p,add,mul,fmas) bind(C, name='fftwf_flops')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), intent(out) :: add
      real(C_DOUBLE), intent(out) :: mul
      real(C_DOUBLE), intent(out) :: fmas
    end subroutine fftwf_flops
    
    real(C_DOUBLE) function fftwf_estimate_cost(p) bind(C, name='fftwf_estimate_cost')
      import
      type(C_PTR), value :: p
    end function fftwf_estimate_cost
    
    real(C_DOUBLE) function fftwf_cost(p) bind(C, name='fftwf_cost')
      import
      type(C_PTR), value :: p
    end function fftwf_cost
    
    integer(C_INT) function fftwf_alignment_of(p) bind(C, name='fftwf_alignment_of')
      import
      real(C_FLOAT), dimension(*), intent(out) :: p
    end function fftwf_alignment_of
    
  end interface
