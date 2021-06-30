<script type="text/javascript">
    (function() {
        // Anonymous function to prevent pollution of global variables
        var utterances = document.createElement('script');
        utterances.type = 'text/javascript';
        utterances.async = true;
        utterances.setAttribute('issue-term','{{ theme.utterance.issue_term }}')
        utterances.setAttribute('theme','{{ theme.utterance.theme }}')
        utterances.setAttribute('repo','{{ theme.utterance.repo }}')
        utterances.crossorigin = 'anonymous';
        utterances.src = 'https://utteranc.es/client.js';
        // content is where you want to insert comments
        document.getElementById('gitment-container').appendChild(utterances);
    })();
</script>
