��   �   0     �    
     |     �  �      ,  �   -  7  �  �  7  -      F   .     u     �  7   �  �   �  �   q  �   $  I      �   j  �   �  �   �  �  ~  H   -     v  E   �  �   <    �  >   �  �   (  9   �  �   �  �   �  �      �   �   �   a!  �   �!  l   �"      #     :#     T#     n#     �#     �#     �#     �#     �#  z   $     �$     �$     �$  .   �$  6   %     ;%     N%     b%     g%  !   }%  !   �%  '   �%     �%     	&     )&  *   H&  /   s&  %   �&     �&  /   �&  ,   '     8'  4   N'  $   �'     �'     �'     �'     �'     (      0(      Q(  h   r(  <   �(     )  :   +)  &   f)  $   �)     �)  2   �)      *  $   *  /   B*  I   r*     �*     �*  3   �*  =   +  6   X+  d   �+  [   �+      P,  O   q,  .   �,  /   �,      -  A   ;-  .   }-  0   �-  )   �-     .     .     ).  ;   C.  <   .  8   �.  6   �.     ,/     I/     d/  (   y/  I   �/  !   �/  '   0  '   60  9   ^0     �0     �0  0   �0     �0  <   �0  -   )1  @   W1  8   �1  >   �1  /   2  7   @2  D   x2  5   �2  &   �2  '   3     B3  %   J3     p3     �3  
   �3  
   �3  
   �3  
   �3  
   �3  	   �3  	   �3  	   �3  	   �3  	   �3  	   �3  	   	4  "   4  *   64     a4  A   u4  Q   �4  *   	5  @   45  !   u5     �5  �  �5  �   �7  �  f8    ;<  9   O>  S   �>     �>     �>  >   ?  �   N?  �   �?  -  �@  M   �A  �   HB    �B  �   �C  3  �D  P   �F  �   6G  F   �G  �   H  >  �H  D   J  �   WJ  6   K  �   NK  �   L  �   �L  #  M  �   5N  �   �N  l   �O     P     0P     LP     fP  "   �P     �P     �P     �P     �P  y   Q     �Q     �Q     �Q  2   �Q  B   R     NR     gR     yR  "   ~R  5   �R  4   �R  <   S  $   IS  $   nS      �S  .   �S  ?   �S  5   #T     YT  9   sT  ;   �T     �T  B   U  )   GU  '   qU  "   �U     �U     �U     �U  $   V  '   >V  p   fV  D   �V     W  7   4W  '   lW  2   �W     �W  6   �W  +   X  4   HX  /   }X  Y   �X     Y     Y  7   5Y  E   mY  <   �Y  y   �Y  a   jZ  6   �Z  Z   [  5   ^[  9   �[  '   �[  B   �[  ?   9\  B   y\  2   �\     �\      �\     ]  ?   7]  I   w]  B   �]  E   ^     J^  #   g^     �^  +   �^  S   �^  &   _  /   E_  /   u_  Q   �_  $   �_     `  P   !`     r`  E   z`  �   �`  G   Fa  D   �a  N   �a  5   "b  4   Xb  N   �b  8   �b  %   c  /   ;c     kc  *   sc     �c     �c     �c     �c     �c     �c     �c     d     d     #d     /d     ;d     Gd     Sd  *   _d  3   �d     �d  `   �d  T   4e  ;   �e  @   �e  &   f     -f     �   V                       l   /       K       	          4   ^   O   �      y   E   �      k   @   o      p   J           M   h   �       $       G       %   |   ]       R       s   ~              e      N   #          &   �      Q   �          b           
   \      6   c       _           Y       "   i   >   .   S                  w          t           u   D       5   r       X               q   �      )      �           !   1   �   A               (       :   C   F   }   B   H   m   +       '   �               f   Z           �      �      �   �                  {   �             �   <   a   �   �   -   *   `   g   n   �   0           ?                   T          z           �       L      �       ,       3   x       �   �   �   �   d   �   j       I   �      [   �   �   9   �   2   7      8   W              U   ;      =   P           v      1f     8f  �  �  �  �  �    ?f         
   ����Vf  2          �����f  0               �����f         
   �����f  F          ����g  2               ���� 
  --delta[=OPTS]      Delta filter; valid OPTS (valid values; default):
                        dist=NUM   distance between bytes being subtracted
                                   from each other (1-256; 1) 
  --lzma1[=OPTS]      LZMA1 or LZMA2; OPTS is a comma-separated list of zero or
  --lzma2[=OPTS]      more of the following options (valid values; default):
                        preset=PRE reset options to a preset (0-9[e])
                        dict=NUM   dictionary size (4KiB - 1536MiB; 8MiB)
                        lc=NUM     number of literal context bits (0-4; 3)
                        lp=NUM     number of literal position bits (0-4; 0)
                        pb=NUM     number of position bits (0-4; 2)
                        mode=MODE  compression mode (fast, normal; normal)
                        nice=NUM   nice length of a match (2-273; 64)
                        mf=NAME    match finder (hc3, hc4, bt2, bt3, bt4; bt4)
                        depth=NUM  maximum search depth; 0=automatic (default) 
  --x86[=OPTS]        x86 BCJ filter (32-bit and 64-bit)
  --powerpc[=OPTS]    PowerPC BCJ filter (big endian only)
  --ia64[=OPTS]       IA-64 (Itanium) BCJ filter
  --arm[=OPTS]        ARM BCJ filter (little endian only)
  --armthumb[=OPTS]   ARM-Thumb BCJ filter (little endian only)
  --sparc[=OPTS]      SPARC BCJ filter
                      Valid OPTS for all BCJ filters:
                        start=NUM  start offset for conversions (default=0) 
 Basic file format and compression options:
 
 Custom filter chain for compression (alternative for using presets): 
 Operation modifiers:
 
 Other options:
 
With no FILE, or when FILE is -, read standard input.
       --block-list=SIZES
                      start a new .xz block after the given comma-separated
                      intervals of uncompressed data       --block-size=SIZE
                      start a new .xz block after every SIZE bytes of input;
                      use this to set the block size for threaded compression       --flush-timeout=TIMEOUT
                      when compressing, if more than TIMEOUT milliseconds has
                      passed since the previous flush and reading more input
                      would block, all pending data is flushed out       --ignore-check  don't verify the integrity check when decompressing       --info-memory   display the total amount of RAM and the currently active
                      memory usage limits, and exit       --memlimit-compress=LIMIT
      --memlimit-decompress=LIMIT
  -M, --memlimit=LIMIT
                      set memory usage limit for compression, decompression,
                      or both; LIMIT is in bytes, % of RAM, or 0 for defaults       --no-adjust     if compression settings exceed the memory usage limit,
                      give an error instead of adjusting the settings downwards       --no-sparse     do not create sparse files when decompressing
  -S, --suffix=.SUF   use the suffix `.SUF' on compressed files
      --files[=FILE]  read filenames to process from FILE; if FILE is
                      omitted, filenames are read from the standard input;
                      filenames must be terminated with the newline character
      --files0[=FILE] like --files but use the null character as terminator       --robot         use machine-parsable messages (useful for scripts)       --single-stream decompress only the first stream, and silently
                      ignore possible remaining input data       CheckVal %*s Header  Flags        CompSize    MemUsage  Filters   -0 ... -9           compression preset; default is 6; take compressor *and*
                      decompressor memory usage into account before using 7-9!   -F, --format=FMT    file format to encode or decode; possible values are
                      `auto' (default), `xz', `lzma', and `raw'
  -C, --check=CHECK   integrity check type: `none' (use with caution),
                      `crc32', `crc64' (default), or `sha256'   -Q, --no-warn       make warnings not affect the exit status   -T, --threads=NUM   use at most NUM threads; the default is 1; set to 0
                      to use as many threads as there are processor cores   -V, --version       display the version number and exit   -e, --extreme       try to improve compression ratio by using more CPU time;
                      does not affect decompressor memory requirements   -h, --help          display the short help (lists only the basic options)
  -H, --long-help     display this long help and exit   -h, --help          display this short help and exit
  -H, --long-help     display the long help (lists also the advanced options)   -k, --keep          keep (don't delete) input files
  -f, --force         force overwrite of output file and (de)compress links
  -c, --stdout        write to standard output and don't delete input files   -q, --quiet         suppress warnings; specify twice to suppress errors too
  -v, --verbose       be verbose; specify twice for even more verbose   -z, --compress      force compression
  -d, --decompress    force decompression
  -t, --test          test compressed file integrity
  -l, --list          list information about .xz files   Blocks:
    Stream     Block      CompOffset    UncompOffset       TotalSize      UncompSize  Ratio  Check   Blocks:             %s
   Check:              %s
   Compressed size:    %s
   Memory needed:      %s MiB
   Minimum XZ Utils version: %s
   Number of files:    %s
   Ratio:              %s
   Sizes in headers:   %s
   Stream padding:     %s
   Streams:
    Stream    Blocks      CompOffset    UncompOffset        CompSize      UncompSize  Ratio  Check      Padding   Streams:            %s
   Uncompressed size:  %s
  Operation mode:
 %s MiB of memory is required. The limit is %s. %s MiB of memory is required. The limiter is disabled. %s file
 %s files
 %s home page: <%s>
 %s:  %s: Cannot remove: %s %s: Cannot set the file group: %s %s: Cannot set the file owner: %s %s: Cannot set the file permissions: %s %s: Closing the file failed: %s %s: Error reading filenames: %s %s: Error seeking the file: %s %s: File already has `%s' suffix, skipping %s: File has setuid or setgid bit set, skipping %s: File has sticky bit set, skipping %s: File is empty %s: File seems to have been moved, not removing %s: Filename has an unknown suffix, skipping %s: Filter chain: %s
 %s: Input file has more than one hard link, skipping %s: Invalid argument to --block-list %s: Invalid filename suffix %s: Invalid multiplier suffix %s: Invalid option name %s: Invalid option value %s: Is a directory, skipping %s: Is a symbolic link, skipping %s: Not a regular file, skipping %s: Null character found when reading filenames; maybe you meant to use `--files0' instead of `--files'? %s: Options must be `name=value' pairs separated with commas %s: Read error: %s %s: Seeking failed when trying to create a sparse file: %s %s: Too many arguments to --block-list %s: Too small to be a valid .xz file %s: Unexpected end of file %s: Unexpected end of input when reading filenames %s: Unknown file format type %s: Unsupported integrity check type %s: Value is not a non-negative decimal integer %s: With --format=raw, --suffix=.SUF is required unless writing to stdout %s: Write error: %s %s: poll() failed: %s --list does not support reading from standard input --list works only on .xz files (--format=xz or --format=auto) 0 can only be used as the last element in --block-list Adjusted LZMA%c dictionary size from %s MiB to %s MiB to not exceed the memory usage limit of %s MiB Adjusted the number of threads from %s to %s to not exceed the memory usage limit of %s MiB Cannot establish signal handlers Cannot read data from standard input when reading filenames from standard input Compressed data cannot be read from a terminal Compressed data cannot be written to a terminal Compressed data is corrupt Compression and decompression with --robot are not supported yet. Compression support was disabled at build time Decompression support was disabled at build time Decompression will need %s MiB of memory. Disabled Empty filename, skipping Error creating a pipe: %s Error getting the file status flags from standard input: %s Error getting the file status flags from standard output: %s Error restoring the O_APPEND flag to standard output: %s Error restoring the status flags to standard input: %s Failed to enable the sandbox File format not recognized Internal error (bug) LZMA1 cannot be used with the .xz format Mandatory arguments to long options are mandatory for short options too.
 Maximum number of filters is four Memory usage limit for compression:     Memory usage limit for decompression:   Memory usage limit is too low for the given filter setup. Memory usage limit reached No No integrity check; not verifying file integrity None Only one file can be specified with `--files' or `--files0'. Report bugs to <%s> (in English or Finnish).
 Strms  Blocks   Compressed Uncompressed  Ratio  Check   Filename Switching to single-threaded mode due to --flush-timeout THIS IS A DEVELOPMENT VERSION NOT INTENDED FOR PRODUCTION USE. The .lzma format supports only the LZMA1 filter The environment variable %s contains too many arguments The exact options of the presets may vary between software versions. The filter chain is incompatible with --flush-timeout The sum of lc and lp must not exceed 4 Total amount of physical memory (RAM):  Totals: Try `%s --help' for more information. Unexpected end of input Unknown error Unknown-11 Unknown-12 Unknown-13 Unknown-14 Unknown-15 Unknown-2 Unknown-3 Unknown-5 Unknown-6 Unknown-7 Unknown-8 Unknown-9 Unsupported LZMA1/LZMA2 preset: %s Unsupported filter chain or filter options Unsupported options Unsupported type of integrity check; not verifying file integrity Usage: %s [OPTION]... [FILE]...
Compress or decompress FILEs in the .xz format.

 Using a preset in raw mode is discouraged. Valid suffixes are `KiB' (2^10), `MiB' (2^20), and `GiB' (2^30). Writing to standard output failed Yes Project-Id-Version: xz 5.2.4
Report-Msgid-Bugs-To: lasse.collin@tukaani.org
PO-Revision-Date: 2021-01-06 22:30-0300
Last-Translator: Rafael Fontenelle <rafaelff@gnome.org>
Language-Team: Brazilian Portuguese <ldpbr-translation@lists.sourceforge.net>
Language: pt_BR
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
X-Bugs: Report translation errors to the Language-Team address.
Plural-Forms: nplurals=2; plural=(n > 1);
X-Generator: Virtaal 1.0.0-beta1
 
  --delta[=OPÇÕES]    filtro delta; OPÇÕES válidas (valores válidos, padrão):
                        dist=NÚM   distância entre bytes sendo subtraído
                                   de cada um (1-256; 1) 
--lzma1[=OPÇÕES]      LZMA1/LZMA2; OPÇÕES é uma lista separada por vírgula de
--lzma2[=OPÇÕES]      zero ou + das opções abaixo (valores válidos, padrão):
                        preset=PRE redefine opções para predefinição (0-9[e])
                        dict=NÚM   tam. de dicionário (4KiB - 1536MiB; 8MiB)
                        lc=NÚM     núm. de bits de contexto literal (0-4; 3)
                        lp=NÚM     núm. de bits de posição literal (0-4; 0)
                        pb=NÚM     núm. de bits de posição (0-4; 2)
                        mode=MODO  modo de compressão (fast, normal; normal)
                        nice=NÚM   tam. de nice de correspondência (2-273; 64)
                        mf=NOME    localizador de correspondência
                                   (hc3, hc4, bt2, bt3, bt4; bt4)
                        depth=NUM  máximo de profundidade de pesquisa;
                                   0=automatic (padrão) 
  --x86[=OPÇÕES]      filtro BCJ x86 (32 bits e 64 bits)
  --powerpc[=OPÇÕES]  filtro BCJ PowerPC (big endian apenas)
  --ia64[=OPÇÕES]     filtro BCJ IA-64 (Itanium)
  --arm[=OPÇÕES]      filtro BCJ ARM (little endian apenas)
  --armthumb[=OPÇÕES] filtro BCJ ARM-Thumb (little endian apenas)
  --sparc[=OPÇÕES]    filtro BCJ SPARC
                      OPÇÕES válidas para todos os filtros BCJ:
                        start=NUM  deslocamento inicial para conversões
                                   (default=0) 
 Opções básicas de formato de arquivo e compressão:
 
 Cadeia de filtros personalizada para compressão (alternativa à predefinição): 
 Modificadores de opções:
 
 Outras opções:
 
Sem ARQUIVO, ou quando ARQUIVO é -, lê da entrada padrão.
       --block-list=TAM
                      inicia um novo bloco .xz após os intervalos dados,
                      separados por vírgula, de dados descomprimidos       --block-size=TAM
                      inicia novo bloco .xz após cada TAM bytes de entrada;
                      use isso para definido o tamanho de bloco para
                      compressão com threads       --flush-timeout=TEMPO-LIMITE
                      ao comprimir, se mais de TEMPO-LIMITE milissegundos
                      tiverem passado desde a liberação anterior e a leitura
                      de mais entrada bloquearia, todos os dados pendentes
                      serão liberados       --ignore-check  não faz a verificação de integridade ao descomprimir       --info-memory   exibe a quantidade total de RAM e os limites de uso
                      de memória atualmente ativos e sai       --memlimit-compress=LIMITE
      --memlimit-decompress=LIMITE
  -M, --memlimit=LIMITE
                      define o limite de uso de memória para compressão,
                      descompressão ou ambos; LIMITE é em bytes, % de RAM
                      ou 0 para padrões       --no-adjust     se configurações de compressão exceder o limite
                      de uso de memória, fornece um erro em vez de
                      ajustar as configurações para baixo       --no-sparse     não cria arquivos esparsos ao descomprimir
  -S, --suffix=.SUF   usa o sufixo ".SUF" em arquivos comprimidos
      --files[=ARQUIVO]
                      lê nomes de arquivos para processar de ARQUIVO;
                      se ARQUIVO for omitido, nomes de arquivos são
                      lidos da entrada padrão; nomes de arquivos devem
                      ser terminados com o caractere de nova linha
      --files0[=ARQUIVO]
                      similar a --files, mas usa o caractere nulo como
                      terminador       --robot         usa mensagens analisáveis por máquina (útil p/ scripts)       --single-stream descomprime apenas o primeiro fluxo, e ignora de forma
                      silenciosa possíveis dados de entrada restantes       ValVerif %*s  Cabeç  Sinaliz       TamComp      UsoMem  Filtros   -0 ... -9           predefinição de compressão; padrão é 6; leve o uso de
                      memória do compressor *e* descompressor em conta
                      antes de usar 7-9!   -F, --format=FMT    formato de arquivo para codificar ou decodificar;
                      valores possíveis são
                      "auto" (padrão), "xz", "lzma" e "raw"
  -C, --check=VERIF   tipo de verificação de integridade: "none" (cuidado!),
                      "crc32", "crc64" (padrão) ou "sha256"   -Q, --no-warn       faz os avisos não afetarem o status de saída   -T, --threads=NÚM   usa no máximo NÚM threads; o padrão é 1; defina para
                      0 para usar o máximo de threads que há de núcleos de
                      processador   -V, --version       exibe o número de versão e sai   -e, --extreme       tenta melhorar a proporção de compressão usando mais
                      tempo de CPU; não afeta os requisitos de memória do
                      descompressor   -h, --help          exibe a ajuda curto (lista apenas as opções básicas)
  -H, --long-help     exibe essa ajuda longa e sai   -h, --help          exibe essa ajuda curta e sai
  -H, --long-help     exibe a ajuda longa (lista também as opções avançadas)   -k, --keep          mantém (não exclui) os arquivos de entrada
  -f, --force         força a sobrescrita do arquivo de entrada e a 
                      (des)compressão de links
  -c, --stdout        escreve a entrada padrão e não exclui os arquivos
                      de entrada   -q, --quiet         suprime avisos, use duas vezes para suprimir erros também
  -v, --verbose       ser detalhado; use duas vezes para ainda mais detalhes   -z, --compress      força a compressão
  -d, --decompress    força a descompressão
  -t, --test          testa a integridade do arquivo comprimido
  -l, --list          lista informações sobre arquivos .xz   Blocos:
    Fluxo      Bloco      DeslocComp   DeslocDescomp    TamanhoTotal   TamanhoDecomp  Propo  Verif   Blocos:             %s
   Verificação:        %s
   Tam. comprimido:    %s
   Memória exigida:    %s MiB
   Versão mínima do XZ Utils: %s
   Núm. de arquivos:   %s
   Proporção:          %s
   Tam. cabeçalhos:    %s
   Ajuste do fluxo:    %s
   Fluxos:
    Fluxo     Blocos      DeslocComp   DeslocDescomp     TamanhoComp  TamanhoDescomp  Propo  Verif       Ajuste   Fluxos:             %s
   Tam. descomprimido: %s
  Modo de operação:
 %s MiB de memória é necessário. O limite é %s. %s MiB de memória é necessário. O limitador está desabilitado. %s arquivo
 %s arquivos
 Site do %s: <%s>
 %s:  %s: Não foi possível remover: %s %s: Não foi possível definir o grupo do arquivo: %s %s: Não foi possível definir o dono do arquivo: %s %s: Não foi possível definir as permissões do arquivo: %s %s: Fechamento do arquivo falhou: %s %s: Erro ao ler nomes de arquivo: %s %s: Erro ao buscar o arquivo: %s %s: O arquivo já tem o sufixo "%s", ignorando %s: O arquivo possui o bit setuid ou setgid definido, ignorando %s: O arquivo possui o bit sticky definido, ignorando %s: O arquivo está vazio %s: O arquivo parece ter sido movido, não será removido %s: O nome de arquivo tem um sufixo desconhecido, ignorando %s: Cadeia de filtros: %s
 %s: O arquivo de entrada possui mais de um link físico, ignorando %s: Argumento inválido para --block-list %s: Sufixo de nome de arquivo inválido %s: Sufixo multiplicador inválido %s: Nome de opção inválido %s: Valor de opção inválido %s: É um diretório, ignorando %s: É um link simbólico, ignorando %s: Não é um arquivo comum, ignorando %s: Caractere nulo encontrado ao ler nomes de arquivos; talvez você queria usar "--files0" em vez de "--files"? %s: As opções devem ser pares "nome=valor" separados por vírgulas %s: Erro de leitura: %s %s: Busca falhou ao tentar criar um arquivo esparso: %s %s: Argumentos demais para --block-list %s: Pequeno demais para ser um arquivo .xz válido %s: Fim de arquivo inesperado %s: Fim da entrada inesperado ao ler nomes de arquivos %s: Tipo de formato de arquivo desconhecido %s: Tipo de verificação de integridade sem suporte %s: O valor não é um inteiro integral decimal %s: Com --format=raw, --suffix=.SUF é exigido, a menos que esteja escrevendo para stdout %s: Erro de escrita: %s %s: poll() falhou: %s --list não possui suporte a leitura da entrada padrão --list funciona apenas em arquivos .xz (--format=xz ou --format=auto) 0 só pode ser usado como o último elemento em --block-list Ajustado o tamanho de dicionário de LZMA%c de %s MiB para %s MiB para não exceder o limite de uso de memória de %s MiB Ajustado o número de threads de %s de %s para não exceder o limite de uso de memória de %s MiB Não foi possível estabelecer manipuladores de sinais Não é possível ler dados da entrada padrão ao ler nomes de arquivos da entrada padrão Dados comprimidos não podem ser lidos de um terminal Dados comprimidos não podem ser escrito para um terminal Os dados comprimidos estão corrompidos Ainda não há suporte a compressão e descompressão com --robot. Suporte a compressão foi desabilitado em tempo de compilação Suporte a descompressão foi desabilitado em tempo de compilação A descompressão precisará de %s MiB de memória. Desabilitado Nome de arquivo vazio, ignorando Erro ao criar um pipe: %s Erro ao obter os sinalizadores de status da entrada padrão: %s Erro ao obter os sinalizadores de status de arquivo da saída padrão: %s Erro ao restaurar o sinalizador O_APPEND para a saída padrão: %s Erro ao restaurar os sinalizadores de status para entrada padrão: %s Falha ao habilitar o sandbox Formato de arquivo não reconhecido Erro interno (bug) LZMA1 não pode ser usado com o formato .xz Argumentos obrigatórios para opções longas também o são para opções curtas.
 O número máximo de filtros é quatro Limite de uso de memória para compressão:     Limite de uso de memória para descompressão:  O limite de uso de memória é baixo demais para a configuração de filtro dada. Limite de uso de memória alcançado Não Sem verificação de integridade; não será verificada a integridade do arquivo Nenhuma Somente um arquivo pode ser especificado com "--files" ou "--files0". Relate erros para <%s> (em inglês ou finlandês).
Relate erros de tradução para <https://translationproject.org/team/pt_BR.html>.
 Fluxos Blocos   Comprimido Descomprimid  Propo  Verif   Nome de Arquivo Alternando para o modo de thread única por causa de --flush-timeout ESSA É UMA VERSÃO DE DESENVOLVIMENTO, NÃO DESTINADA PARA USO EM PRODUÇÃO. O formato .lzma possui suporte apenas ao filtro LZMA1 A variável de ambiente %s contém argumentos demais As opções exatas de predefinições podem variar entre versões do software. A cadeia de filtros é incompatível com --flush-timeout A soma de lc e lp não deve exceder 4 Quantidade total de memória física (RAM):     Totais: Tente "%s --help" para mais informações. Fim da entrada inesperado Erro desconhecido Incógnito11 Incógnito12 Incógnito13 Incógnito14 Incógnito15 Incógnito2 Incógnito3 Incógnito5 Incógnito6 Incógnito7 Incógnito8 Incógnito9 Predefinição LZMA1/LZMA2 sem suporte: %s Opções de filtro ou cadeia de filtros sem suporte Opções sem suporte Tipo de verificação de integridade sem suporte; não será verificada a integridade do arquivo Uso: %s [OPÇÕES]... [ARQUIVO]...
Comprime e descomprime ARQUIVOs no formato .xz.

 O uso de uma predefinição em modo bruto é desencorajado. Sufixos válidos são "KiB" (2^10), "MiB" (2^20) e "GiB" (2^30). A escrita para a saída padrão falhou Sim PRIu32 PRIu64 Using up to % threads. The selected match finder requires at least nice=% Value of the option `%s' must be in the range [%, %] Usando até % threads. O localizador de correspondência selecionado requer pelo menos nice=% O valor da opção "%s" deve estar no intervalo [%, %] 