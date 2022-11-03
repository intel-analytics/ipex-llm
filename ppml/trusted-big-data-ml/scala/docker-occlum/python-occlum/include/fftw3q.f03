! Generated automatically.  DO NOT EDIT!


  type, bind(C) :: fftwq_iodim
     integer(C_INT) n, is, os
  end type fftwq_iodim
  type, bind(C) :: fftwq_iodim64
     integer(C_INTPTR_T) n, is, os
  end type fftwq_iodim64

  interface
    type(C_PTR) function fftwq_plan_dft(rank,n,in,out,sign,flags) bind(C, name='fftwq_plan_dft')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      complex(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwq_plan_dft
    
    type(C_PTR) function fftwq_plan_dft_1d(n,in,out,sign,flags) bind(C, name='fftwq_plan_dft_1d')
      import
      integer(C_INT), value :: n
      complex(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_1d
    
    type(C_PTR) function fftwq_plan_dft_2d(n0,n1,in,out,sign,flags) bind(C, name='fftwq_plan_dft_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      complex(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_2d
    
    type(C_PTR) function fftwq_plan_dft_3d(n0,n1,n2,in,out,sign,flags) bind(C, name='fftwq_plan_dft_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      complex(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_3d
    
    type(C_PTR) function fftwq_plan_many_dft(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,sign,flags) &
                         bind(C, name='fftwq_plan_many_dft')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      complex(16), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwq_plan_many_dft
    
    type(C_PTR) function fftwq_plan_guru_dft(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
                         bind(C, name='fftwq_plan_guru_dft')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim), dimension(*), intent(in) :: howmany_dims
      complex(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwq_plan_guru_dft
    
    type(C_PTR) function fftwq_plan_guru_split_dft(rank,dims,howmany_rank,howmany_dims,ri,ii,ro,io,flags) &
                         bind(C, name='fftwq_plan_guru_split_dft')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: ri
      real(16), dimension(*), intent(out) :: ii
      real(16), dimension(*), intent(out) :: ro
      real(16), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwq_plan_guru_split_dft
    
    type(C_PTR) function fftwq_plan_guru64_dft(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
                         bind(C, name='fftwq_plan_guru64_dft')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim64), dimension(*), intent(in) :: howmany_dims
      complex(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: sign
      integer(C_INT), value :: flags
    end function fftwq_plan_guru64_dft
    
    type(C_PTR) function fftwq_plan_guru64_split_dft(rank,dims,howmany_rank,howmany_dims,ri,ii,ro,io,flags) &
                         bind(C, name='fftwq_plan_guru64_split_dft')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim64), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: ri
      real(16), dimension(*), intent(out) :: ii
      real(16), dimension(*), intent(out) :: ro
      real(16), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwq_plan_guru64_split_dft
    
    subroutine fftwq_execute_dft(p,in,out) bind(C, name='fftwq_execute_dft')
      import
      type(C_PTR), value :: p
      complex(16), dimension(*), intent(inout) :: in
      complex(16), dimension(*), intent(out) :: out
    end subroutine fftwq_execute_dft
    
    subroutine fftwq_execute_split_dft(p,ri,ii,ro,io) bind(C, name='fftwq_execute_split_dft')
      import
      type(C_PTR), value :: p
      real(16), dimension(*), intent(inout) :: ri
      real(16), dimension(*), intent(inout) :: ii
      real(16), dimension(*), intent(out) :: ro
      real(16), dimension(*), intent(out) :: io
    end subroutine fftwq_execute_split_dft
    
    type(C_PTR) function fftwq_plan_many_dft_r2c(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) &
                         bind(C, name='fftwq_plan_many_dft_r2c')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      real(16), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: flags
    end function fftwq_plan_many_dft_r2c
    
    type(C_PTR) function fftwq_plan_dft_r2c(rank,n,in,out,flags) bind(C, name='fftwq_plan_dft_r2c')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      real(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_r2c
    
    type(C_PTR) function fftwq_plan_dft_r2c_1d(n,in,out,flags) bind(C, name='fftwq_plan_dft_r2c_1d')
      import
      integer(C_INT), value :: n
      real(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_r2c_1d
    
    type(C_PTR) function fftwq_plan_dft_r2c_2d(n0,n1,in,out,flags) bind(C, name='fftwq_plan_dft_r2c_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      real(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_r2c_2d
    
    type(C_PTR) function fftwq_plan_dft_r2c_3d(n0,n1,n2,in,out,flags) bind(C, name='fftwq_plan_dft_r2c_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      real(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_r2c_3d
    
    type(C_PTR) function fftwq_plan_many_dft_c2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) &
                         bind(C, name='fftwq_plan_many_dft_c2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      complex(16), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_INT), value :: flags
    end function fftwq_plan_many_dft_c2r
    
    type(C_PTR) function fftwq_plan_dft_c2r(rank,n,in,out,flags) bind(C, name='fftwq_plan_dft_c2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      complex(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_c2r
    
    type(C_PTR) function fftwq_plan_dft_c2r_1d(n,in,out,flags) bind(C, name='fftwq_plan_dft_c2r_1d')
      import
      integer(C_INT), value :: n
      complex(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_c2r_1d
    
    type(C_PTR) function fftwq_plan_dft_c2r_2d(n0,n1,in,out,flags) bind(C, name='fftwq_plan_dft_c2r_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      complex(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_c2r_2d
    
    type(C_PTR) function fftwq_plan_dft_c2r_3d(n0,n1,n2,in,out,flags) bind(C, name='fftwq_plan_dft_c2r_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      complex(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_dft_c2r_3d
    
    type(C_PTR) function fftwq_plan_guru_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwq_plan_guru_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_guru_dft_r2c
    
    type(C_PTR) function fftwq_plan_guru_dft_c2r(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwq_plan_guru_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim), dimension(*), intent(in) :: howmany_dims
      complex(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_guru_dft_c2r
    
    type(C_PTR) function fftwq_plan_guru_split_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,ro,io,flags) &
                         bind(C, name='fftwq_plan_guru_split_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: ro
      real(16), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwq_plan_guru_split_dft_r2c
    
    type(C_PTR) function fftwq_plan_guru_split_dft_c2r(rank,dims,howmany_rank,howmany_dims,ri,ii,out,flags) &
                         bind(C, name='fftwq_plan_guru_split_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: ri
      real(16), dimension(*), intent(out) :: ii
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_guru_split_dft_c2r
    
    type(C_PTR) function fftwq_plan_guru64_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwq_plan_guru64_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim64), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: in
      complex(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_guru64_dft_r2c
    
    type(C_PTR) function fftwq_plan_guru64_dft_c2r(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
                         bind(C, name='fftwq_plan_guru64_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim64), dimension(*), intent(in) :: howmany_dims
      complex(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_guru64_dft_c2r
    
    type(C_PTR) function fftwq_plan_guru64_split_dft_r2c(rank,dims,howmany_rank,howmany_dims,in,ro,io,flags) &
                         bind(C, name='fftwq_plan_guru64_split_dft_r2c')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim64), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: ro
      real(16), dimension(*), intent(out) :: io
      integer(C_INT), value :: flags
    end function fftwq_plan_guru64_split_dft_r2c
    
    type(C_PTR) function fftwq_plan_guru64_split_dft_c2r(rank,dims,howmany_rank,howmany_dims,ri,ii,out,flags) &
                         bind(C, name='fftwq_plan_guru64_split_dft_c2r')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim64), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: ri
      real(16), dimension(*), intent(out) :: ii
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), value :: flags
    end function fftwq_plan_guru64_split_dft_c2r
    
    subroutine fftwq_execute_dft_r2c(p,in,out) bind(C, name='fftwq_execute_dft_r2c')
      import
      type(C_PTR), value :: p
      real(16), dimension(*), intent(inout) :: in
      complex(16), dimension(*), intent(out) :: out
    end subroutine fftwq_execute_dft_r2c
    
    subroutine fftwq_execute_dft_c2r(p,in,out) bind(C, name='fftwq_execute_dft_c2r')
      import
      type(C_PTR), value :: p
      complex(16), dimension(*), intent(inout) :: in
      real(16), dimension(*), intent(out) :: out
    end subroutine fftwq_execute_dft_c2r
    
    subroutine fftwq_execute_split_dft_r2c(p,in,ro,io) bind(C, name='fftwq_execute_split_dft_r2c')
      import
      type(C_PTR), value :: p
      real(16), dimension(*), intent(inout) :: in
      real(16), dimension(*), intent(out) :: ro
      real(16), dimension(*), intent(out) :: io
    end subroutine fftwq_execute_split_dft_r2c
    
    subroutine fftwq_execute_split_dft_c2r(p,ri,ii,out) bind(C, name='fftwq_execute_split_dft_c2r')
      import
      type(C_PTR), value :: p
      real(16), dimension(*), intent(inout) :: ri
      real(16), dimension(*), intent(inout) :: ii
      real(16), dimension(*), intent(out) :: out
    end subroutine fftwq_execute_split_dft_c2r
    
    type(C_PTR) function fftwq_plan_many_r2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,kind,flags) &
                         bind(C, name='fftwq_plan_many_r2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      integer(C_INT), value :: howmany
      real(16), dimension(*), intent(out) :: in
      integer(C_INT), dimension(*), intent(in) :: inembed
      integer(C_INT), value :: istride
      integer(C_INT), value :: idist
      real(16), dimension(*), intent(out) :: out
      integer(C_INT), dimension(*), intent(in) :: onembed
      integer(C_INT), value :: ostride
      integer(C_INT), value :: odist
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwq_plan_many_r2r
    
    type(C_PTR) function fftwq_plan_r2r(rank,n,in,out,kind,flags) bind(C, name='fftwq_plan_r2r')
      import
      integer(C_INT), value :: rank
      integer(C_INT), dimension(*), intent(in) :: n
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwq_plan_r2r
    
    type(C_PTR) function fftwq_plan_r2r_1d(n,in,out,kind,flags) bind(C, name='fftwq_plan_r2r_1d')
      import
      integer(C_INT), value :: n
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind
      integer(C_INT), value :: flags
    end function fftwq_plan_r2r_1d
    
    type(C_PTR) function fftwq_plan_r2r_2d(n0,n1,in,out,kind0,kind1,flags) bind(C, name='fftwq_plan_r2r_2d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind0
      integer(C_FFTW_R2R_KIND), value :: kind1
      integer(C_INT), value :: flags
    end function fftwq_plan_r2r_2d
    
    type(C_PTR) function fftwq_plan_r2r_3d(n0,n1,n2,in,out,kind0,kind1,kind2,flags) bind(C, name='fftwq_plan_r2r_3d')
      import
      integer(C_INT), value :: n0
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), value :: kind0
      integer(C_FFTW_R2R_KIND), value :: kind1
      integer(C_FFTW_R2R_KIND), value :: kind2
      integer(C_INT), value :: flags
    end function fftwq_plan_r2r_3d
    
    type(C_PTR) function fftwq_plan_guru_r2r(rank,dims,howmany_rank,howmany_dims,in,out,kind,flags) &
                         bind(C, name='fftwq_plan_guru_r2r')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwq_plan_guru_r2r
    
    type(C_PTR) function fftwq_plan_guru64_r2r(rank,dims,howmany_rank,howmany_dims,in,out,kind,flags) &
                         bind(C, name='fftwq_plan_guru64_r2r')
      import
      integer(C_INT), value :: rank
      type(fftwq_iodim64), dimension(*), intent(in) :: dims
      integer(C_INT), value :: howmany_rank
      type(fftwq_iodim64), dimension(*), intent(in) :: howmany_dims
      real(16), dimension(*), intent(out) :: in
      real(16), dimension(*), intent(out) :: out
      integer(C_FFTW_R2R_KIND), dimension(*), intent(in) :: kind
      integer(C_INT), value :: flags
    end function fftwq_plan_guru64_r2r
    
    subroutine fftwq_execute_r2r(p,in,out) bind(C, name='fftwq_execute_r2r')
      import
      type(C_PTR), value :: p
      real(16), dimension(*), intent(inout) :: in
      real(16), dimension(*), intent(out) :: out
    end subroutine fftwq_execute_r2r
    
    subroutine fftwq_destroy_plan(p) bind(C, name='fftwq_destroy_plan')
      import
      type(C_PTR), value :: p
    end subroutine fftwq_destroy_plan
    
    subroutine fftwq_forget_wisdom() bind(C, name='fftwq_forget_wisdom')
      import
    end subroutine fftwq_forget_wisdom
    
    subroutine fftwq_cleanup() bind(C, name='fftwq_cleanup')
      import
    end subroutine fftwq_cleanup
    
    subroutine fftwq_set_timelimit(t) bind(C, name='fftwq_set_timelimit')
      import
      real(C_DOUBLE), value :: t
    end subroutine fftwq_set_timelimit
    
    subroutine fftwq_plan_with_nthreads(nthreads) bind(C, name='fftwq_plan_with_nthreads')
      import
      integer(C_INT), value :: nthreads
    end subroutine fftwq_plan_with_nthreads
    
    integer(C_INT) function fftwq_planner_nthreads() bind(C, name='fftwq_planner_nthreads')
      import
    end function fftwq_planner_nthreads
    
    integer(C_INT) function fftwq_init_threads() bind(C, name='fftwq_init_threads')
      import
    end function fftwq_init_threads
    
    subroutine fftwq_cleanup_threads() bind(C, name='fftwq_cleanup_threads')
      import
    end subroutine fftwq_cleanup_threads
    
! Unable to generate Fortran interface for fftwq_threads_set_callback
    subroutine fftwq_make_planner_thread_safe() bind(C, name='fftwq_make_planner_thread_safe')
      import
    end subroutine fftwq_make_planner_thread_safe
    
    integer(C_INT) function fftwq_export_wisdom_to_filename(filename) bind(C, name='fftwq_export_wisdom_to_filename')
      import
      character(C_CHAR), dimension(*), intent(in) :: filename
    end function fftwq_export_wisdom_to_filename
    
    subroutine fftwq_export_wisdom_to_file(output_file) bind(C, name='fftwq_export_wisdom_to_file')
      import
      type(C_PTR), value :: output_file
    end subroutine fftwq_export_wisdom_to_file
    
    type(C_PTR) function fftwq_export_wisdom_to_string() bind(C, name='fftwq_export_wisdom_to_string')
      import
    end function fftwq_export_wisdom_to_string
    
    subroutine fftwq_export_wisdom(write_char,data) bind(C, name='fftwq_export_wisdom')
      import
      type(C_FUNPTR), value :: write_char
      type(C_PTR), value :: data
    end subroutine fftwq_export_wisdom
    
    integer(C_INT) function fftwq_import_system_wisdom() bind(C, name='fftwq_import_system_wisdom')
      import
    end function fftwq_import_system_wisdom
    
    integer(C_INT) function fftwq_import_wisdom_from_filename(filename) bind(C, name='fftwq_import_wisdom_from_filename')
      import
      character(C_CHAR), dimension(*), intent(in) :: filename
    end function fftwq_import_wisdom_from_filename
    
    integer(C_INT) function fftwq_import_wisdom_from_file(input_file) bind(C, name='fftwq_import_wisdom_from_file')
      import
      type(C_PTR), value :: input_file
    end function fftwq_import_wisdom_from_file
    
    integer(C_INT) function fftwq_import_wisdom_from_string(input_string) bind(C, name='fftwq_import_wisdom_from_string')
      import
      character(C_CHAR), dimension(*), intent(in) :: input_string
    end function fftwq_import_wisdom_from_string
    
    integer(C_INT) function fftwq_import_wisdom(read_char,data) bind(C, name='fftwq_import_wisdom')
      import
      type(C_FUNPTR), value :: read_char
      type(C_PTR), value :: data
    end function fftwq_import_wisdom
    
    subroutine fftwq_fprint_plan(p,output_file) bind(C, name='fftwq_fprint_plan')
      import
      type(C_PTR), value :: p
      type(C_PTR), value :: output_file
    end subroutine fftwq_fprint_plan
    
    subroutine fftwq_print_plan(p) bind(C, name='fftwq_print_plan')
      import
      type(C_PTR), value :: p
    end subroutine fftwq_print_plan
    
    type(C_PTR) function fftwq_sprint_plan(p) bind(C, name='fftwq_sprint_plan')
      import
      type(C_PTR), value :: p
    end function fftwq_sprint_plan
    
    type(C_PTR) function fftwq_malloc(n) bind(C, name='fftwq_malloc')
      import
      integer(C_SIZE_T), value :: n
    end function fftwq_malloc
    
    type(C_PTR) function fftwq_alloc_real(n) bind(C, name='fftwq_alloc_real')
      import
      integer(C_SIZE_T), value :: n
    end function fftwq_alloc_real
    
    type(C_PTR) function fftwq_alloc_complex(n) bind(C, name='fftwq_alloc_complex')
      import
      integer(C_SIZE_T), value :: n
    end function fftwq_alloc_complex
    
    subroutine fftwq_free(p) bind(C, name='fftwq_free')
      import
      type(C_PTR), value :: p
    end subroutine fftwq_free
    
    subroutine fftwq_flops(p,add,mul,fmas) bind(C, name='fftwq_flops')
      import
      type(C_PTR), value :: p
      real(C_DOUBLE), intent(out) :: add
      real(C_DOUBLE), intent(out) :: mul
      real(C_DOUBLE), intent(out) :: fmas
    end subroutine fftwq_flops
    
    real(C_DOUBLE) function fftwq_estimate_cost(p) bind(C, name='fftwq_estimate_cost')
      import
      type(C_PTR), value :: p
    end function fftwq_estimate_cost
    
    real(C_DOUBLE) function fftwq_cost(p) bind(C, name='fftwq_cost')
      import
      type(C_PTR), value :: p
    end function fftwq_cost
    
    integer(C_INT) function fftwq_alignment_of(p) bind(C, name='fftwq_alignment_of')
      import
      real(16), dimension(*), intent(out) :: p
    end function fftwq_alignment_of
    
  end interface
